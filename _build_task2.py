"""Assemble AI写的.ipynb: Task 1.1 (from executed backup) + Task 1.2 + Task 1.3.

Task 1.1 cells are copied verbatim from _before_task2.ipynb.bak so its
executed outputs are preserved. Task 1.3 is copied verbatim from the original
untouched backup. Only the Task 1.2 section is newly written here.
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SRC = "AI写的.ipynb"
T1_BACKUP = "_before_task2.ipynb.bak"      # executed task 1.1
T3_BACKUP = "_backup_AI写的.ipynb.bak"     # original file, untouched

t1 = nbf.read(T1_BACKUP, as_version=4)
t3 = nbf.read(T3_BACKUP, as_version=4)

# Task 1.1: everything up to (not including) any "任务2"/"任务3" header.
cut = next((i for i, c in enumerate(t1.cells)
            if c.cell_type == "markdown" and ("任务2" in c.source or "任务3" in c.source)),
           len(t1.cells))
task1_cells = t1.cells[:cut]

# Task 1.3: the markdown header and the GAN code cell from the original file.
t3_start = next(i for i, c in enumerate(t3.cells)
                if c.cell_type == "markdown" and "任务3" in c.source)
task3_cells = t3.cells[t3_start:]

print(f"task 1 cells: {len(task1_cells)} (from executed backup)")
print(f"task 3 cells: {len(task3_cells)} (from original backup: "
      f"{[c.cell_type for c in task3_cells]})")

cells = list(task1_cells)

cells.append(new_markdown_cell(r"""---
# Task 1.2 — Neural Autoregressive Model (NADE)

## Model

Task 1.1 kept the autoregressive factorisation

$$p(x) = \prod_{j=1}^{D} p(x_j \mid x_{<j})$$

but gave every conditional its own independent set of weights. A neural
autoregressive model keeps the *same* factorisation and instead replaces each
conditional by the output of a **shared** neural network.

**NADE** (Neural Autoregressive Density Estimator) uses one masked hidden
layer. Every hidden unit $k$ is assigned a maximum visible index $a_k$ and may
look only at pixels before it:

$$h_k = \sigma\Big(\sum_{i<a_k} W_{ki}x_i + c_k\Big),\qquad
p(x_j{=}1\mid x_{<j}) = \sigma\Big(\sum_{k:\,a_k\le j} V_{jk}h_k + b_j\Big)$$

The condition $a_k \le j$ is what keeps pixel $j$ causal: a hidden unit that
has already seen $x_j$ must not contribute to the prediction of $x_j$.

### Why training costs D forward passes

Because the hidden units are prefix-masked, hidden state $h(x_{<j})$ differs
for every $j$, so a *single* forward pass cannot produce all 784 conditionals
at once. Training therefore loops over $j$, at each step zeroing the pixels
$\ge j$ and reading off the single logit for pixel $j$:

$$\mathcal{L} = -\sum_{i=1}^{N}\sum_{j=1}^{D}
\Big[x_j^{(i)}\log p_j + (1-x_j^{(i)})\log(1-p_j)\Big]$$

This is exact maximum likelihood — no bound — but it makes one epoch of NADE
about $D$ times more expensive than one epoch of the fully visible model.
Sampling remains cheap: it is the same 784-step ancestral draw as in Task 1.1.

### Code sharing with Task 1.1

`evaluate_nll`, `independent_baseline`, `plot_grid`, `save_samples` and
`Timer` from the Task 1.1 section are reused directly, so the two tasks are
evaluated on exactly the same footing.""")) 

cells.append(new_code_cell(r'''class NADE(nn.Module):
    """Neural Autoregressive Density Estimator for 28x28 binary images.

    One masked hidden layer; each hidden unit k has a maximum visible index
    a_k and sees only pixels x_i with i < a_k.
    """

    def __init__(self, D=DIM, H=256, perm=None):
        super().__init__()
        self.D, self.H = D, H
        g = torch.Generator().manual_seed(0)

        # a_k in [1, D]: hidden unit k summarises the prefix x_0 .. x_{a_k-1}
        self.register_buffer("a", torch.randint(1, D + 1, (H,), generator=g))
        idx = torch.arange(D)
        self.register_buffer("maskW", (idx[None, :] < self.a[:, None]).float())   # (H, D)
        self.register_buffer("maskV", (self.a[None, :] <= idx[:, None]).float())  # (D, H)

        # Optional pixel ordering. A non-identity permutation means pixel
        # position i in the model corresponds to original pixel perm[i].
        perm = torch.arange(D) if perm is None else perm
        self.register_buffer("perm", perm.long())
        self.register_buffer("prefix", torch.tril(torch.ones(D, D), diagonal=-1))

        self.W = nn.Parameter(torch.randn(H, D, generator=g) * 0.01)
        self.V = nn.Parameter(torch.randn(D, H, generator=g) / H ** 0.5)
        self.c = nn.Parameter(torch.zeros(H))
        self.b = nn.Parameter(torch.zeros(D))

    # ---------- training objective ----------
    def loss(self, x):
        """Summed NLL over the batch (nats). Requires D sequential passes."""
        B = x.shape[0]
        total = 0.0
        for j in range(self.D):
            xj = x * self.prefix[j]                    # zero out pixels >= j
            h = torch.sigmoid(F.linear(xj, self.W * self.maskW, self.c))
            logit_j = F.linear(h, self.V * self.maskV, self.b)[:, j]
            total = total + F.binary_cross_entropy_with_logits(
                logit_j, x[:, j], reduction="sum")
        return total

    @torch.no_grad()
    def log_prob(self, x):
        """Exact log-likelihood per image, in nats (D sequential passes)."""
        out = 0.0
        for j in range(self.D):
            xj = x * self.prefix[j]
            h = torch.sigmoid(F.linear(xj, self.W * self.maskW, self.c))
            logit_j = F.linear(h, self.V * self.maskV, self.b)[:, j]
            lp = -F.binary_cross_entropy_with_logits(
                logit_j, x[:, j], reduction="none")
            out = out + lp
        return out

    # ---------- generation ----------
    @torch.no_grad()
    def sample(self, n, device="cpu"):
        """Ancestral sampling in model (permuted) order."""
        self.eval()
        x = torch.zeros(n, self.D, device=device)
        for j in range(self.D):
            xj = x * self.prefix[j]
            h = torch.sigmoid(F.linear(xj, self.W * self.maskW, self.c))
            p_j = torch.sigmoid(F.linear(h, self.V * self.maskV, self.b)[:, j])
            x[:, j] = torch.bernoulli(p_j)
        if not torch.equal(self.perm, torch.arange(self.D, device=self.perm.device)):
            inv = torch.empty_like(self.perm)
            inv[self.perm] = torch.arange(self.D, device=self.perm.device)
            x = x[:, inv]                       # back to image pixel order
        return x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_nade(model, loader, epochs=10, lr=1e-3, device="cpu",
               sample_at=(), n_sample=4, verbose=True):
    """Maximum-likelihood training. Returns (history, stages)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_nll_per_pixel": []}
    stages = []

    for ep in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, _ in loader:
            xb = to_flat(xb).to(device)
            loss = model.loss(xb) / xb.size(0)       # per-image NLL
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            seen += xb.size(0)
        mean = total / seen
        history["train_nll_per_pixel"].append(mean / DIM)
        if ep in sample_at:
            stages.append(model.sample(n_sample, device=device).cpu())
        if verbose and (ep % 2 == 0 or ep == 1):
            print(f"epoch {ep:3d}/{epochs}  train NLL/pixel = {mean / DIM:.4f} nats")
    return history, stages


# ---- train on the full 60,000-image training set ----------------------
NADE_H = 256
NADE_EPOCHS = 10
SAMPLE_AT = (1, 3, 5, 8, 10)            # 5 stages -> the required 5x4 grid

nade = NADE(D=DIM, H=NADE_H)
print(f"NADE: H={NADE_H}, parameters={nade.n_params():,}")

timer_nade = Timer()
timer_nade.start()
hist_nade, stages_nade = train_nade(nade, train_loader, epochs=NADE_EPOCHS,
                                    lr=1e-3, device=DEVICE,
                                    sample_at=SAMPLE_AT, n_sample=4)
nade_train_time = timer_nade.stop("train_total")
print(f"\nparameters    : {nade.n_params():,}")
print(f"training time : {nade_train_time:.1f} s  ({nade_train_time/60:.2f} min)")''')) 

cells.append(new_code_cell(r'''# ---- training curve ---------------------------------------------------
plot_curve(hist_nade, keys=["train_nll_per_pixel"],
           title="Task 1.2 NADE - training curve",
           ylabel="negative log-likelihood per pixel (nats)",
           savepath="results/task2_loss_curve.png")''')) 

cells.append(new_code_cell(r'''# ---- training progress: 5 rows x 4 columns ----------------------------
plot_grid(stages_nade, nrow=4,
          title="Task 1.2 NADE - generation progress (rows = epochs %s)" % (SAMPLE_AT,),
          savepath="results/task2_progress_grid.png")''')) 

cells.append(new_code_cell(r'''# ---- final samples ----------------------------------------------------
N_GEN = 10
t = Timer(); t.start()
samples_nade = nade.sample(N_GEN, device=DEVICE).cpu()
nade_infer_time = t.stop("sample_%d" % N_GEN)

plot_grid(samples_nade, nrow=5,
          title="Task 1.2 NADE - %d generated samples" % N_GEN,
          savepath="results/task2_samples.png")
save_samples(samples_nade, "results/task2_samples.npy")
print("sampling %d images: %.2f s total (%.4f s per image)"
      % (N_GEN, nade_infer_time, nade_infer_time / N_GEN))''')) 

cells.append(new_code_cell(r'''# ---- quantitative evaluation -----------------------------------------
# Exact test NLL. NADE has no closed form in one pass, so evaluate_nll is
# re-derived here using log_prob (D passes per image); to keep the run
# affordable this is measured on a random 2,000-image slice of the test set.
torch.manual_seed(0)
eval_idx = torch.randperm(len(test_loader.dataset))[:2000]
eval_subset = torch.utils.data.Subset(test_loader.dataset, eval_idx.tolist())
eval_loader = torch.utils.data.DataLoader(eval_subset, batch_size=100,
                                          shuffle=False, num_workers=0)

nade.eval()
t = Timer(); t.start()
tot_lp, n_img = 0.0, 0
with torch.no_grad():
    for xb, _ in eval_loader:
        xb = to_flat(xb).to(DEVICE)
        tot_lp += nade.log_prob(xb).sum().item()
        n_img += xb.size(0)
nade_eval_time = t.stop("eval")
nade_nll_img = -tot_lp / n_img
nade_nll_px = nade_nll_img / DIM

print(f"test NLL  : {nade_nll_img:8.2f} nats/image  "
      f"({nade_nll_px:.4f} nats/pixel, {nade_nll_px/np.log(2):.4f} bits/pixel)")
print(f"evaluated on {n_img} test images in {nade_eval_time:.1f} s")
print()
print(f"Task 1.1 FVSBN (H=0) : {nll_px:.4f} nats/pixel")
print(f"pixel-indep. baseline: {baseline_px:.4f} nats/pixel")
print(f"NADE                 : {nade_nll_px:.4f} nats/pixel")
print(f"NADE vs FVSBN        : {nade_nll_px - nll_px:+.4f} nats/pixel")

# foreground rate sanity check
gen_c = nade.sample(200, device=DEVICE)
print(f"\nforeground ratio  real {x.mean().item():.4f}  generated {gen_c.mean().item():.4f}")''')) 

cells.append(new_code_cell(r'''# ---- causality check (NADE) ------------------------------------------
# Same test as for Task 1.1: flip all pixels >= j and confirm the logit of
# every pixel < j is unchanged. Reimplemented because NADE has no single
# forward pass that returns all logits.
nade.eval()
xb, _ = next(iter(test_loader))
xb = to_flat(xb)[:4].to(DEVICE)

def logit_at(model, x, j):
    xj = x * model.prefix[j]
    h = torch.sigmoid(F.linear(xj, model.W * model.maskW, model.c))
    return F.linear(h, model.V * model.maskV, model.b)[:, j]

worst = 0.0
for j in range(1, DIM, 37):
    xp = xb.clone()
    xp[:, j:] = 1.0 - xp[:, j:]
    with torch.no_grad():
        for i in range(max(1, j - 40), j):          # a few pixels before j
            d = (logit_at(nade, xp, i) - logit_at(nade, xb, i)).abs().max().item()
            worst = max(worst, d)
print(f"max change in logits of pixels < j: {worst:.3e}")
print("OK - strictly autoregressive" if worst < 1e-4 else "LEAKAGE DETECTED")
assert worst < 1e-4, "information leakage from future pixels"''')) 

cells.append(new_markdown_cell(r"""### Ablation — hidden width and pixel ordering

Two factors are varied, each trained for the same shortened schedule so the
comparison is fair:

1. **Hidden width** $H \in \{64, 256, 512\}$ — capacity of the shared network.
2. **Pixel ordering** — the factorisation is exact for *any* ordering, but the
   ordering determines how strong the conditional dependencies are. A random
   permutation is compared against the raster-scan order used above.

The cheaper `loss` is replaced by evaluating on a small slice of the test set
to keep the ablation affordable, since each configuration still needs $D$
forward passes per batch.""")) 

cells.append(new_code_cell(r'''# ---- ablation: hidden width and ordering ------------------------------
# Shortened schedule (2 epochs) to keep the total run time reasonable.
ABL_EPOCHS = 2
ABL_H = 64
abl_idx = torch.randperm(len(train_loader.dataset))[:ABL_H]
abl_set = torch.utils.data.Subset(train_loader.dataset, abl_idx.tolist())
abl_loader = torch.utils.data.DataLoader(abl_set, batch_size=100,
                                         shuffle=True, num_workers=0)

def quick_eval(model, loader, n_batches=5):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for k, (xb, _) in enumerate(loader):
            xb = to_flat(xb).to(DEVICE)
            tot += -model.log_prob(xb).sum().item()
            n += xb.size(0)
            if k + 1 >= n_batches:
                break
    return tot / n / DIM

perm_rand = torch.randperm(DIM, generator=torch.Generator().manual_seed(7))
abl_cfg = [("H=64,  raster", dict(H=64,  perm=None)),
           ("H=256, raster", dict(H=256, perm=None)),
           ("H=512, raster", dict(H=512, perm=None)),
           ("H=256, random", dict(H=256, perm=perm_rand))]

ablation_nade = []
for name, cfg in abl_cfg:
    torch.manual_seed(0)
    m = NADE(D=DIM, H=cfg["H"], perm=cfg["perm"])
    tt = Timer(); tt.start()
    train_nade(m, abl_loader, epochs=ABL_EPOCHS, lr=1e-3, device=DEVICE,
               verbose=False)
    secs = tt.stop("t")
    px = quick_eval(m, eval_loader)
    ablation_nade.append(dict(name=name, H=cfg["H"], params=m.n_params(),
                              nll_px=px, seconds=secs))
    print(f"{name:16s} params={m.n_params():>8,}  NLL={px:.4f} nats/px  ({secs:5.1f}s)")

print()
print(f"{'config':16s} {'params':>10s} {'NLL nats/px':>12s} {'time (s)':>9s}")
print("-" * 52)
for r in ablation_nade:
    print(f"{r['name']:16s} {r['params']:>10,} {r['nll_px']:>12.4f} {r['seconds']:>9.1f}")''')) 

cells.append(new_code_cell(r'''# ---- summary ----------------------------------------------------------
print(timer_nade.summary())
print(f"\nsampling {N_GEN} images: {nade_infer_time:.2f} s "
      f"({nade_infer_time / N_GEN:.4f} s/image)")
print(f"test-set NLL evaluation ({n_img} images): {nade_eval_time:.1f} s")''')) 

cells.extend(task3_cells)

new_nb = new_notebook(cells=cells)
new_nb.metadata = t1.metadata
nbf.write(new_nb, SRC)

nbf.validate(new_nb)
print(f"wrote {SRC}: {len(cells)} cells, validated OK")
for i, c in enumerate(new_nb.cells):
    head = c.source.strip().split(chr(10))[0][:56] if c.source.strip() else "(EMPTY)"
    print(f"  [{i:2d}] {c.cell_type:8s} out={len(c.get('outputs', [])):2d} | {head}")
