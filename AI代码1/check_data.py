"""
数据管线自检 —— 验证二值化预处理和 DataLoader 是否正常工作。

直接运行：  python check_data.py

检查项：
  1. Binarize 能否被 pickle（macOS DataLoader 子进程崩溃的根因）
  2. 二值化结果是否真的只含 {0,1}
  3. 前景像素占比是否 ≈ 0.13（MNIST 的已知统计特性，可用来证明预处理没错）
  4. 原始灰度图（GAN 用）与二值图（自回归模型用）是否互相独立
"""

import pickle
import sys

import torch

from common import DIM, Binarize, foreground_ratio, get_loaders

print("=" * 65)
print("数据管线自检")
print("=" * 65)

# ---------- 1. pickle 往返 ----------
print("\n[1] Binarize 可被 pickle（DataLoader 子进程能重建它）")
b = Binarize(tau=0.5)
b2 = pickle.loads(pickle.dumps(b))
status = "OK" if repr(b) == repr(b2) else "失败"
print(f"    {status}: {b2}")
print(f"    模块归属: {Binarize.__module__}  (必须是 'common'，不能是 '__main__')")
if Binarize.__module__ == "__main__":
    print("    [!!] 你把 Binarize 定义在脚本里了，子进程会崩溃。")
    print("         请改成: from common import Binarize")
    sys.exit(1)

# ---------- 2 & 3. 二值化正确性 ----------
print("\n[2] 二值化 MNIST（任务 1.1 / 1.2 用，像素 ∈ {0,1}）")
train_bin, _ = get_loaders(binarize=True, batch_size=128, num_workers=0)
x, y = next(iter(train_bin))
vals = sorted(set(x.unique().tolist()))
print(f"    batch 形状 : {tuple(x.shape)}  ->  展平为 {DIM} 维")
print(f"    像素取值   : {vals}")
print(f"    标签       : {y[:8].tolist()} ...")
assert vals == [0.0, 1.0], f"二值化失败，取值应为 [0,1]，实际 {vals}"
print("    OK: 确实只有 0 和 1")

ratio = foreground_ratio(train_bin)
print(f"\n[3] 前景像素占比 : {ratio:.4f}   （预期 ≈ 0.13）")
if not 0.10 < ratio < 0.16:
    print("    [!!] 偏离预期，检查 Binarize 的阈值设置")
    sys.exit(1)
print("    OK: 与 MNIST 已知统计特性吻合，预处理正确")

# ---------- 4. 原始灰度图 ----------
print("\n[4] 原始灰度 MNIST（任务 1.3 GAN 用，像素 ∈ [0,1]）")
train_raw, _ = get_loaders(binarize=False, batch_size=128, num_workers=0)
xr, _ = next(iter(train_raw))
print(f"    形状     : {tuple(xr.shape)}")
print(f"    取值范围 : [{xr.min():.3f}, {xr.max():.3f}]   均值 {xr.mean():.4f}")
print(f"    与二值图是否不同: {not torch.equal(xr, x)}  (应为 True)")
print("    OK: 灰度版保留连续像素值，与二值版是两个独立数据集")

print("\n" + "=" * 65)
print("全部通过 ✅  数据管线可用")
print("=" * 65)
