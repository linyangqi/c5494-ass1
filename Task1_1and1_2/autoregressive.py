"""Task 1.1 FVSBN and Task 1.2 MADE: exact Bernoulli likelihoods.

Pixels have degrees 1..D in raster order. A hidden unit with degree m can
read inputs of degree <= m; output j can read hidden degrees strictly < j.
Thus neither the target pixel nor a future pixel can reach its own logit.
Reference: Germain et al., ICML 2015, https://proceedings.mlr.press/v37/germain15.html
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, mask: torch.Tensor, bias: bool = True):
        super().__init__(mask.shape[1], mask.shape[0], bias=bias)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveModel(nn.Module):
    """Shared interface. NLL is a vector, one exact value per image."""
    def nll(self, x):
        return F.binary_cross_entropy_with_logits(self(x), x, reduction="none").sum(1)

    @torch.inference_mode()
    def sample(self, n: int, seed: int = 2026):
        """Ancestral samples with a local RNG; training RNG is unaffected."""
        self.eval()
        device = next(self.parameters()).device
        rng = torch.Generator(device=device).manual_seed(seed)
        # The same uniform draws across checkpoints make progress comparisons fair.
        uniform = torch.rand(n, self.dim, generator=rng, device=device)
        x = torch.zeros_like(uniform)
        for j in range(self.dim):
            p = self(x)[:, j].sigmoid()
            x[:, j] = (uniform[:, j] < p).float()
        return x

    def parameter_counts(self):
        """Allocated parameters vs. active connections (masked weights don't learn)."""
        total = sum(p.numel() for p in self.parameters())
        active = total
        for m in self.modules():
            if isinstance(m, MaskedLinear):
                active -= m.weight.numel() - int(m.mask.sum().item())
        return {"allocated": total, "active": active}


class FVSBN(AutoregressiveModel):
    """p(x_j=1|x_<j) = sigmoid(b_j + sum_{i<j} W_ji x_i)."""
    def __init__(self, dim=784):
        super().__init__()
        self.dim = dim
        self.linear = MaskedLinear(torch.tril(torch.ones(dim, dim), diagonal=-1))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    @torch.inference_mode()
    def initialize_bias(self, probabilities):
        self.linear.bias.copy_(torch.logit(probabilities.clamp(1e-4, 1 - 1e-4)))

    @torch.inference_mode()
    def sample(self, n, seed=2026):
        # Only compute the requested row, rather than D full output vectors.
        self.eval()
        device = self.linear.weight.device
        rng = torch.Generator(device=device).manual_seed(seed)
        uniform = torch.rand(n, self.dim, generator=rng, device=device)
        x = torch.zeros_like(uniform)
        for j in range(self.dim):
            logit = F.linear(x[:, :j], self.linear.weight[j:j + 1, :j],
                             self.linear.bias[j:j + 1]).squeeze(1)
            x[:, j] = (uniform[:, j] < logit.sigmoid()).float()
        return x


class MADE(AutoregressiveModel):
    """Masked MLP; optional strictly causal linear skip is an explicit variant.

    Fixed, evenly spread hidden degrees give every layer coverage of the pixel
    ordering. ReLU hidden units and unconstrained output logits are used; BCE
    applies the sigmoid stably. There is no sequential loop during training.
    """
    def __init__(self, dim=784, hidden=(512, 512), residual=False, mask_seed=0):
        super().__init__()
        if dim < 2 or not hidden or any(h < 1 for h in hidden):
            raise ValueError("MADE requires dim >= 2 and nonempty positive hidden widths")
        self.dim = dim
        self.hidden = tuple(hidden)
        self.residual = residual
        rng = torch.Generator().manual_seed(mask_seed)
        previous = torch.arange(1, dim + 1)
        layers = []
        for width in hidden:
            degree = torch.linspace(1, dim - 1, width).round().long()
            degree = degree[torch.randperm(width, generator=rng)]
            layer = MaskedLinear(degree[:, None] >= previous[None, :])
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.ReLU()])
            previous = degree
        self.hidden_net = nn.Sequential(*layers)
        self.output = MaskedLinear(torch.arange(1, dim + 1)[:, None] > previous[None, :])
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        if residual:
            self.skip = MaskedLinear(torch.tril(torch.ones(dim, dim), diagonal=-1), bias=False)
            nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        logits = self.output(self.hidden_net(x))
        return logits + self.skip(x) if self.residual else logits

    @torch.inference_mode()
    def initialize_bias(self, probabilities):
        self.output.bias.copy_(torch.logit(probabilities.clamp(1e-4, 1 - 1e-4)))


def build_model(config):
    if config["model"] == "fvsbn":
        return FVSBN(config.get("dim", 784))
    if config["model"] == "made":
        return MADE(config.get("dim", 784), tuple(config["hidden"]),
                    config.get("residual", False), config.get("mask_seed", 0))
    raise ValueError(f"Unknown model: {config['model']}")


def check_causality(model, dim=None):
    """Check target AND future leakage, including both boundary pixels."""
    dim = dim or model.dim
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(41)
    x = torch.rand(3, dim, generator=rng, device=device)
    pivots = sorted(set([0, 1, dim // 2, dim - 1] + list(range(0, dim, max(1, dim // 17)))))
    worst = 0.0
    with torch.no_grad():
        base = model(x)
        for j in pivots:
            changed = x.clone()
            changed[:, j:] = 1 - changed[:, j:]
            # Include output j: its own input must not influence its prediction.
            worst = max(worst, (model(changed)[:, :j + 1] - base[:, :j + 1]).abs().max().item())
    if worst > 1e-6:
        raise AssertionError(f"Autoregressive leakage: {worst}")
    return {"max_logit_change": worst, "pivots_checked": len(pivots)}
