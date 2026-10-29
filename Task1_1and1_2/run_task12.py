"""Train, evaluate and export the task 1.1/1.2 experiments.

Run from any directory: python run_task12.py
Completed runs are reused only when config AND data fingerprints match.
Use --refresh to regenerate the comparisons without retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import zipfile
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from autoregressive import build_model, check_causality

OUT = ROOT / "results" / "task12"
SPLIT_SEED = 5494
SCHEMA = 1
COLORS = ["#176B87", "#DA7B27", "#578F63", "#8C61A5", "#D85B66", "#717B89"]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_artifacts():
    """Restore bundled data/results on first use; never overwrite existing files.

    Extract only paths inside this folder. The bundle is portable: no dependency
    on the original workspace or sibling directories is needed.
    """
    bundle = ROOT / "artifacts.zip"
    if not bundle.exists():
        return 0
    count = 0
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if member.is_dir() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def find_data(root=None):
    ensure_artifacts()
    if root:
        return Path(root).resolve()
    candidates = sorted(ROOT.glob("**/MNIST/raw/train-images-idx3-ubyte"))
    if candidates:
        return candidates[0].parent.parent.parent
    # Portable fallback for a fresh Colab/checkout. Existing data is reused.
    from torchvision.datasets import MNIST
    path = ROOT / "data"
    MNIST(str(path), train=True, download=True)
    MNIST(str(path), train=False, download=True)
    return path


def read_images(path):
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = np.frombuffer(f.read(), dtype=np.uint8).copy()
    if magic != 2051 or (rows, cols) != (28, 28) or raw.size != count * 784:
        raise ValueError(f"Invalid MNIST image file: {path}")
    # Exactly equivalent to ToTensor() > 0.5: uint8 values >= 128 become 1.
    return torch.from_numpy((raw.reshape(count, 784) >= 128).astype(np.float32))


def load_data(data_root=None):
    root = find_data(data_root)
    raw = root / "MNIST" / "raw"
    train_file, test_file = raw / "train-images-idx3-ubyte", raw / "t10k-images-idx3-ubyte"
    all_train, test = read_images(train_file), read_images(test_file)
    if len(all_train) != 60000 or len(test) != 10000:
        raise ValueError("Expected the official MNIST 60000/10000 split")
    idx = torch.randperm(60000, generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_idx, val_idx = idx[:55000], idx[55000:]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "split_indices.npz", train=train_idx.numpy(), validation=val_idx.numpy())
    info = {"train": 55000, "validation": 5000, "test": 10000,
            "split_seed": SPLIT_SEED, "threshold": "uint8 >= 128 (float > 0.5)",
            "pixel_order": "28x28 raster scan, top-left to bottom-right",
            "train_sha256": file_sha256(train_file), "test_sha256": file_sha256(test_file)}
    info["fingerprint"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    write_json(OUT / "dataset.json", info)
    return all_train[train_idx], all_train[val_idx], test, info


def configs(epochs=40):
    common = {"dim": 784, "epochs": epochs, "batch_size": 256, "lr": 0.001,
              "seed": 0, "mask_seed": 0, "train_size": 55000,
              "weight_decay": 0.0, "schema": SCHEMA}
    return [dict(common, name="fvsbn", model="fvsbn", hidden=[], residual=False),
            dict(common, name="made_256", model="made", hidden=[256], residual=False),
            dict(common, name="made_512", model="made", hidden=[512], residual=False),
            dict(common, name="made_512x2", model="made", hidden=[512, 512], residual=False),
            dict(common, name="made_512x2_skip", model="made", hidden=[512, 512], residual=True),
            dict(common, name="made_512x2_10k", model="made", hidden=[512, 512], residual=False,
                 train_size=10000)]


@torch.inference_mode()
def evaluate(model, x, batch_size=1024):
    model.eval()
    values = torch.cat([model.nll(b) for b in x.split(batch_size)]).cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite likelihood")
    return {"nll_image": float(values.mean()), "nll_pixel": float(values.mean() / 784),
            "bits_pixel": float(values.mean() / (784 * np.log(2))),
            "standard_error_image": float(values.std(ddof=1) / np.sqrt(len(values))),
            "n": len(values)}, values


def plot_samples(samples, path, title, cols=8, row_labels=None):
    images = np.asarray(samples).reshape(-1, 28, 28)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05 + .4, rows * 1.05 + .5),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < len(images):
            ax.imshow(images[k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row_labels and k % cols == 0:
            ax.set_ylabel(row_labels[k // cols], fontsize=10, rotation=0, labelpad=34, va="center")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(.06 if row_labels else 0, 0, 1, .97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history, path, name, best_epoch):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot([r["epoch"] for r in history], [r["train_nll_image"] / 784 for r in history],
            label="Training (online epoch mean)", color=COLORS[0])
    ax.plot([r["epoch"] for r in history], [r["val_nll_image"] / 784 for r in history],
            label="Validation (epoch-end)", color=COLORS[1])
    ax.axvline(best_epoch, color="#777777", linestyle="--", linewidth=1, label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="NLL (nats / pixel)", title=f"{name}: learning curves")
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def train_run(cfg, train, val, data_info, threads):
    path = OUT / cfg["name"]
    path.mkdir(parents=True, exist_ok=True)
    completion = path / "training.json"
    if completion.exists():
        saved = read_json(completion)
        if saved["config"] != cfg or saved["data_fingerprint"] != data_info["fingerprint"]:
            raise ValueError(f"Existing run differs: {path}; choose another --output directory")
        if not (path / "best.pt").exists():
            raise FileNotFoundError(path / "best.pt")
        print(f"[reuse] {cfg['name']}", flush=True)
        return saved

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    x = train[:cfg["train_size"]]
    model.initialize_bias(x.mean(0))
    causal = check_causality(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Same epoch budget/schedule for all capacity and dataset-size comparisons.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["lr"] * .1)
    rng = torch.Generator().manual_seed(cfg["seed"] + 123)
    sample_epochs = sorted(set([1, max(1, cfg["epochs"] // 4), max(1, cfg["epochs"] // 2),
                                max(1, cfg["epochs"] * 3 // 4), cfg["epochs"]]))
    history, stages = [], []
    best, best_epoch = float("inf"), 0
    fit_seconds = validation_seconds = snapshot_seconds = 0.0
    start = time.perf_counter()
    print(f"[train] {cfg['name']} train={len(x)} epochs={cfg['epochs']} threads={threads}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        t = time.perf_counter()
        model.train()
        order = torch.randperm(len(x), generator=rng)
        total = 0.0
        for indices in order.split(cfg["batch_size"]):
            batch = x[indices]
            loss = model.nll(batch).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {cfg['name']} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total += loss.item() * len(batch)
        fit_seconds += time.perf_counter() - t
        t = time.perf_counter()
        metrics, _ = evaluate(model, val)
        validation_seconds += time.perf_counter() - t
        row = {"epoch": epoch, "train_nll_image": total / len(x),
               "val_nll_image": metrics["nll_image"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if metrics["nll_image"] < best:
            best, best_epoch = metrics["nll_image"], epoch
            torch.save({"config": cfg, "model_state": model.state_dict(), "epoch": epoch,
                        "val_nll_image": best, "data_fingerprint": data_info["fingerprint"]}, path / "best.pt")
        if epoch in sample_epochs:
            t = time.perf_counter()
            stages.append(model.sample(4, seed=2026).numpy())
            snapshot_seconds += time.perf_counter() - t
        scheduler.step()
        write_json(path / "history.json", history)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg["epochs"]:
            print(f"  epoch {epoch:3d}: train={row['train_nll_image']:.2f} val={metrics['nll_image']:.2f} "
                  f"nats/image elapsed={time.perf_counter() - start:.1f}s", flush=True)

    wall = time.perf_counter() - start
    np.save(path / "progress.npy", np.stack(stages))
    plot_samples(np.concatenate(stages), path / "progress.png", f"{cfg['name']}: fixed-randomness progress",
                 cols=4, row_labels=[f"epoch {ep}" for ep in sample_epochs])
    plot_history(history, path / "loss_curve.png", cfg["name"], best_epoch)
    result = {"config": cfg, "data_fingerprint": data_info["fingerprint"],
              "best_epoch": best_epoch, "best_val_nll_image": best,
              "fit_seconds": fit_seconds, "validation_seconds": validation_seconds,
              "snapshot_seconds": snapshot_seconds, "training_wall_seconds": wall,
              "sample_epochs": sample_epochs, "causality": causal,
              "parameters": model.parameter_counts(), "threads": threads}
    write_json(completion, result)
    return result


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def finalize(runs, train, val, test, data_info):
    # Select architecture using VALIDATION ONLY, before opening any test metric.
    candidates = [r for r in runs if r["config"]["model"] == "made"
                  and r["config"]["train_size"] == 55000 and r["config"]["seed"] == 0]
    winner = min(candidates, key=lambda r: r["best_val_nll_image"])
    selection = {"task1": "fvsbn", "task2": winner["config"]["name"],
                 "rule": "lowest validation NLL among full-data seed-0 MADE configurations",
                 "test_used_for_selection": False}
    write_json(OUT / "selection.json", selection)

    # Fit the independent model on TRAIN, evaluate on the same held-out TEST.
    p = train.mean(0).clamp(1e-4, 1 - 1e-4)
    base_values = F.binary_cross_entropy(p.expand_as(test), test, reduction="none").sum(1).numpy()
    baseline = {"name": "independent", "test_nll_image": float(base_values.mean()),
                "test_nll_pixel": float(base_values.mean() / 784),
                "test_standard_error_image": float(base_values.std(ddof=1) / np.sqrt(len(test))),
                "fit_split": "train (55000)", "test_size": len(test)}
    np.save(OUT / "independent_test_nll.npy", base_values)
    write_json(OUT / "baseline.json", baseline)
    rows = []
    for run in runs:
        name = run["config"]["name"]
        path = OUT / name
        print(f"[evaluate/export] {name}", flush=True)
        model, ckpt = load_checkpoint(path / "best.pt")
        causal = check_causality(model)
        test_metrics, per_image = evaluate(model, test)
        train_metrics, _ = evaluate(model, train[:run["config"]["train_size"]])
        np.save(path / "test_nll.npy", per_image)
        # Fixed seed, first 64 draws: no filtering or selection of attractive digits.
        t = time.perf_counter()
        samples = model.sample(64, seed=2026).numpy()
        sample64_s = time.perf_counter() - t
        np.save(path / "samples.npy", samples)
        plot_samples(samples, path / "samples.png", f"{name}: 64 uncurated samples")
        plot_samples(samples[:10], path / "samples_10.png", f"{name}: first 10 samples", cols=5)
        # Warm-up + three repeats, same batch size for all models.
        model.sample(10, seed=7)
        times = []
        for seed in (7, 8, 9):
            t = time.perf_counter(); model.sample(10, seed=seed)
            times.append(time.perf_counter() - t)
        result = dict(run, train=train_metrics, test=test_metrics, causality=causal,
                      checkpoint_sha256=file_sha256(path / "best.pt"), sample64_seconds=sample64_s,
                      sample10_seconds_median=float(np.median(times)), sample10_seconds_repeats=times,
                      generated_foreground_ratio=float(samples.mean()),
                      test_foreground_ratio=float(test.mean()),
                      unique_samples_64=int(np.unique(samples, axis=0).shape[0]))
        write_json(path / "metrics.json", result)
        rows.append({"name": name, "model": run["config"]["model"],
                     "hidden": "x".join(map(str, run["config"]["hidden"])) or "none",
                     "residual": run["config"]["residual"], "train_size": run["config"]["train_size"],
                     "seed": run["config"]["seed"], "best_epoch": run["best_epoch"],
                     "active_params": run["parameters"]["active"],
                     "allocated_params": run["parameters"]["allocated"],
                     "val_nll_image": run["best_val_nll_image"],
                     "train_nll_image": train_metrics["nll_image"],
                     "test_nll_image": test_metrics["nll_image"],
                     "test_nll_pixel": test_metrics["nll_pixel"],
                     "test_bits_pixel": test_metrics["bits_pixel"],
                     "test_standard_error_image": test_metrics["standard_error_image"],
                     "fit_seconds": run["fit_seconds"], "training_wall_seconds": run["training_wall_seconds"],
                     "sample10_seconds": result["sample10_seconds_median"]})
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(OUT / "comparison.json", rows)

    selected_nll = np.load(OUT / selection["task2"] / "test_nll.npy")
    fvsbn_nll = np.load(OUT / "fvsbn" / "test_nll.npy")
    difference = selected_nll - fvsbn_nll
    se = difference.std(ddof=1) / np.sqrt(len(difference))
    write_json(OUT / "paired_comparison.json", {
        "difference": "selected MADE minus FVSBN, nats/image; negative favors MADE",
        "mean": float(difference.mean()), "approx_95pct_ci": [float(difference.mean()-1.96*se), float(difference.mean()+1.96*se)],
        "n": len(difference), "note": "Sampling uncertainty over test images, not over training seeds."})
    plot_comparison(rows, baseline)
    return selection


def plot_comparison(rows, baseline):
    primary = [r for r in rows if r["seed"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = [r["name"].replace("made_", "MADE ").replace("_", "\n") for r in primary]
    x = np.arange(len(primary))
    axes[0].bar(x, [r["test_nll_image"] for r in primary], color=COLORS[:len(primary)],
                yerr=[1.96*r["test_standard_error_image"] for r in primary], capsize=3)
    axes[0].set(ylabel="Test NLL (nats/image; lower is better)", title="Held-out likelihood (95% image-level CI)")
    axes[0].text(.98, .97, f"Independent baseline: {baseline['test_nll_image']:.1f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9)
    axes[1].bar(x, [r["training_wall_seconds"] / 60 for r in primary], color=COLORS[:len(primary)])
    axes[1].set(ylabel="Training + validation + snapshots (minutes)", title="Measured CPU run time")
    for ax in axes:
        ax.set_xticks(x, names, fontsize=8); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=160); plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--only", nargs="+", help="Train named configurations only; no test evaluation")
    parser.add_argument("--refresh", action="store_true", help="Export existing complete runs only")
    parser.add_argument("--repeat-seeds", action="store_true", help="Also train validation-selected MADE at seeds 1 and 2")
    args = parser.parse_args()
    ensure_artifacts()
    OUT = args.output.resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    environment_path = OUT / ("environment.json" if not (OUT / "environment.json").exists() else "last_invocation.json")
    write_json(environment_path, {
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "device": "cpu", "threads": args.threads,
        "deterministic_algorithms": True, "command": sys.argv,
        "source_sha256": {f: file_sha256(ROOT/f) for f in ("autoregressive.py", "run_task12.py")}})
    train, val, test, info = load_data(args.data_root)
    all_cfg = configs(args.epochs)
    if args.only:
        unknown = set(args.only) - {c["name"] for c in all_cfg}
        if unknown:
            raise ValueError(f"Unknown configurations {unknown}")
        for cfg in all_cfg:
            if cfg["name"] in args.only:
                train_run(cfg, train, val, info, args.threads)
        return
    if args.refresh:
        runs = [read_json(OUT / c["name"] / "training.json") for c in all_cfg]
        runs += [read_json(p) for p in sorted(OUT.glob("*_seed*/training.json"))]
    else:
        runs = [train_run(cfg, train, val, info, args.threads) for cfg in all_cfg]
        if args.repeat_seeds:
            winner = min([r for r in runs if r["config"]["model"] == "made" and r["config"]["train_size"] == 55000],
                         key=lambda r: r["best_val_nll_image"])
            for seed in (1, 2):
                cfg = dict(winner["config"], seed=seed, name=winner["config"]["name"] + f"_seed{seed}")
                runs.append(train_run(cfg, train, val, info, args.threads))
    selected = finalize(runs, train, val, test, info)
    print("DONE", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
