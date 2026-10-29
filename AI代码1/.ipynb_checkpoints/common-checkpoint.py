"""
CS5494 Assignment 1 -- shared utility module for all three tasks.

Why does this file exist?
  1. On macOS / Jupyter, DataLoader worker processes (num_workers > 0) use
     pickle to rebuild objects by "module name + class name". If a class is
     defined inside a notebook cell, its module is '__main__', which the
     worker cannot resolve, producing:
         AttributeError: Can't get attribute 'Binarize' on <module '__main__'>
     Putting the class in this .py file makes it a normally importable module,
     which removes the problem at its root.
  2. Data loading, binarization, plotting and timing are identical across the
     three tasks, so they are written once here to avoid duplication.

Usage in all three tasks:
    from common import get_device, get_loaders, plot_grid, plot_curve, Timer
"""

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data directory: always the data/ folder next to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "data")
RESULTS_DIR = os.path.join(_HERE, "results")

# Official MNIST statistics (used for Normalize)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
IMG_SIZE = 28
DIM = IMG_SIZE * IMG_SIZE          # 784, the flattened dimension


# ======================================================================
# 1. Device selection
# ======================================================================
def get_device(prefer=None, verbose=True):
    """Return a torch.device.

    prefer: None -> automatic (cuda > mps > cpu); may also force
            'cpu' / 'mps' / 'cuda'

    NOTE: torch 2.2.x has a known defect in torch.bernoulli on Apple MPS.
    When sampling from the autoregressive models (tasks 1.1 / 1.2), move the
    probabilities to CPU first, or simply pass prefer='cpu'. Training itself
    on MPS is fine.
    """
    if prefer is not None:
        device = torch.device(prefer)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[device] 使用 {device}")
        if device.type == "mps":
            print("[device] 提示：采样阶段若报错或出 NaN，改用 prefer='cpu'")
    return device


def bernoulli_cpu(probs):
    """Draw Bernoulli samples on CPU, then move the result back.

    Works around the known torch.bernoulli problem on MPS in torch 2.2.x.
    probs:  float tensor on any device, entries in [0, 1]
    return: 0/1 float tensor, same device and shape as probs
    """
    device = probs.device
    out = torch.bernoulli(probs.detach().cpu().float())
    return out.to(device)


# ======================================================================
# 2. Data loading
# ======================================================================
class Binarize:
    """Convert a [0,1] grayscale image into a {0,1} binary image
    (used by tasks 1.1 / 1.2).

    tau:        threshold, default 0.5. MNIST pixel intensities are strongly
                bimodal, so 0.5 is the natural choice; it keeps the
                foreground ratio at about 0.13 (useful as a sanity check).
    stochastic: True  -> binarize by Bernoulli sampling, which matches the
                         modelling assumption that each pixel is Bernoulli
                         distributed, but the same image differs across reads;
                False -> hard threshold, deterministic and reproducible.
    """

    def __init__(self, tau=0.5, stochastic=False):
        self.tau = tau
        self.stochastic = stochastic

    def __call__(self, x):
        # x: (1, 28, 28) float in [0,1]
        if self.stochastic:
            return torch.bernoulli(x)
        return (x > self.tau).float()

    def __repr__(self):
        return f"Binarize(tau={self.tau}, stochastic={self.stochastic})"


def get_datasets(binarize=False, tau=0.5, stochastic=False, root=DATA_ROOT):
    """Return (train_dataset, test_dataset).

    binarize=False -> original grayscale, pixels in [0,1]  (task 1.3, GAN)
    binarize=True  -> binary images, pixels in {0,1}       (tasks 1.1 / 1.2)
    """
    if binarize:
        tf = transforms.Compose([transforms.ToTensor(), Binarize(tau, stochastic)])
    else:
        tf = transforms.ToTensor()

    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return train, test


def get_loaders(binarize=False, batch_size=128, num_workers=0, tau=0.5,
                stochastic=False, subset=None, root=DATA_ROOT, seed=0):
    """Return (train_loader, test_loader).

    subset: None uses all 60,000 training images; an integer draws a random
            subset of that size (the assignment requires that the subset size
            be stated in the report if a subset is used).
    num_workers: default 0. On macOS / Jupyter this must be 0, otherwise the
            worker processes crash because custom classes such as Binarize
            cannot be pickled. MNIST is tiny, so 0 is fast enough.
    """
    train, test = get_datasets(binarize, tau, stochastic, root)

    if subset is not None and subset < len(train):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train), generator=g)[:subset].tolist()
        train = torch.utils.data.Subset(train, idx)

    # With num_workers > 0, persistent_workers helps, but the default is 0
    common = dict(batch_size=batch_size, num_workers=num_workers)
    if num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train, shuffle=True, drop_last=False, **common)
    test_loader = DataLoader(test, shuffle=False, drop_last=False, **common)
    return train_loader, test_loader


def to_flat(x):
    """(B,1,28,28) -> (B,784), for fully connected autoregressive models."""
    return x.view(x.size(0), -1)


def foreground_ratio(loader, n_batches=8):
    """Measure the foreground (white pixel) ratio; after binarizing MNIST it
    should be about 0.13.

    This is the quickest sanity check that the binarization is correct.
    """
    total, count = 0.0, 0
    for i, (x, _) in enumerate(loader):
        total += x.sum().item()
        count += x.numel()
        if i + 1 >= n_batches:
            break
    return total / count


# ======================================================================
# 3. Visualization
# ======================================================================
def _ensure_results():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def to_grid_image(samples, nrow=4, pad=2):
    """Tile a batch of (N,1,28,28) or (N,784) samples into one large image
    and return it as a numpy array of shape (H,W).

    Does not rely on torchvision.utils.make_grid, avoiding version differences.
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu()
    samples = samples.float()
    if samples.dim() == 2:
        samples = samples.view(-1, 1, IMG_SIZE, IMG_SIZE)

    n = samples.size(0)
    ncol = nrow
    nrows = (n + ncol - 1) // ncol

    H = nrows * IMG_SIZE + (nrows + 1) * pad
    W = ncol * IMG_SIZE + (ncol + 1) * pad
    canvas = np.zeros((H, W), dtype=np.float32)

    for i in range(n):
        r, c = divmod(i, ncol)
        y = pad + r * (IMG_SIZE + pad)
        x = pad + c * (IMG_SIZE + pad)
        canvas[y:y + IMG_SIZE, x:x + IMG_SIZE] = samples[i, 0].numpy()

    return canvas


def plot_grid(samples, nrow=4, title=None, savepath=None, cmap="gray",
              vmin=0.0, vmax=1.0, figsize=None):
    """Plot a grid of generated samples. `samples` may be a single batch of
    shape (N,1,28,28), or a training-progress sequence (a list of batches).

    For the progress grid (the 5 rows x 4 columns layout explicitly required
    by item 2.3 of the assignment):
        plot_grid(list_of_sample_batches, nrow=4, savepath='results/progress.png')
    Each element of the list is one batch saved at a training stage, and each
    stage is drawn as one row.
    """
    import matplotlib.pyplot as plt

    _ensure_results()

    # --- Case A: training-progress sequence -> one row per stage ---
    if isinstance(samples, (list, tuple)) and len(samples) > 0 and torch.is_tensor(samples[0]):
        stages = samples
        nrow = stages[0].size(0)
        rows = [to_grid_image(s, nrow=nrow) for s in stages]
        row_h = rows[0].shape[0]
        total_h = len(rows) * row_h
        total_w = rows[0].shape[1]
        canvas = np.zeros((total_h, total_w), dtype=np.float32)
        for i, r in enumerate(rows):
            canvas[i * row_h:(i + 1) * row_h, :r.shape[1]] = r

        if figsize is None:
            figsize = (total_w / 40, total_h / 40)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        # Label each stage on the left
        for i in range(len(rows)):
            ax.text(-6, i * row_h + row_h / 2, f"stage {i+1}",
                    ha="right", va="center", fontsize=10)
        if title:
            ax.set_title(title, fontsize=13)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"[plot_grid] 已保存 {savepath}")
        plt.show()
        return fig

    # --- Case B: a single batch of samples ---
    canvas = to_grid_image(samples, nrow=nrow)
    if figsize is None:
        figsize = (canvas.shape[1] / 40, canvas.shape[0] / 40)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[plot_grid] 已保存 {savepath}")
    plt.show()
    return fig


def plot_curve(history, keys=None, title="Training curve", xlabel="epoch",
               ylabel="loss", savepath=None, logy=False):
    """Plot training curves. `history` is a dict, e.g.
    {'train_loss': [...], 'val_loss': [...]}.

    Note: pass the mean loss per epoch, not the loss of the last batch,
    otherwise the curve is jagged and unusable in the report.
    """
    import matplotlib.pyplot as plt

    _ensure_results()
    if keys is None:
        keys = list(history.keys())

    fig, ax = plt.subplots(figsize=(7, 4))
    for k in keys:
        if k in history and len(history[k]) > 0:
            ax.plot(range(1, len(history[k]) + 1), history[k], label=k, linewidth=1.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[plot_curve] 已保存 {savepath}")
    plt.show()
    return fig


def save_samples(samples, path):
    """Save samples to a .npy file so plots can be redrawn without retraining."""
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    np.save(path, samples)
    print(f"[save_samples] 已保存 {path}  形状={samples.shape}")


# ======================================================================
# 4. Timing (item 2.3 requires reporting training and inference time)
# ======================================================================
class Timer:
    """Usage:
        t = Timer(); t.start()
        ... training ...
        t.stop('train_total')
        print(t.summary())
    """

    def __init__(self):
        self.records = {}
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()
        return self

    def stop(self, name):
        if self._t0 is None:
            raise RuntimeError("请先调用 start()")
        elapsed = time.perf_counter() - self._t0
        self.records[name] = elapsed
        self._t0 = None
        return elapsed

    @staticmethod
    def now():
        return time.perf_counter()

    def add(self, name, seconds):
        self.records[name] = seconds
        return seconds

    def get(self, name):
        return self.records.get(name)

    def summary(self):
        if not self.records:
            return "(无计时记录)"
        width = max(len(k) for k in self.records)
        lines = ["计时结果："]
        for k, v in self.records.items():
            lines.append(f"  {k:<{width}} : {v:8.2f} 秒  ({v/60:.2f} 分钟)")
        return "\n".join(lines)


def format_seconds(s):
    """Format a number of seconds into a human-readable string."""
    if s < 60:
        return f"{s:.2f} 秒"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)} 分 {sec:.1f} 秒"
    h, m = divmod(int(m), 60)
    return f"{h} 小时 {m} 分 {sec:.1f} 秒"
