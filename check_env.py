"""
环境自检脚本 —— 确认 PyTorch 是否可用、用哪个设备训练、MNIST 数据是否就绪。

用法：
  1. 命令行：  python check_env.py
  2. notebook：把本文件内容粘进一格，或 !python check_env.py

如果 import torch 失败，说明还没安装，请在终端运行：
  conda install pytorch torchvision -c pytorch -y
装完记得重启 Jupyter kernel。
"""

import sys
import platform
import os

print("=" * 60)
print("环境自检")
print("=" * 60)

# ---------- 1. Python 解释器 ----------
print(f"Python 版本 : {sys.version.split()[0]}")
print(f"解释器路径  : {sys.executable}")
print(f"系统        : {platform.system()} {platform.machine()}")

# ---------- 2. NumPy ----------
# torch 2.2.2 是针对 NumPy 1.x 编译的，配 NumPy 2.x 会导致
# "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"
# 并且让 torch <-> numpy 的转换（画图、算 metric）全部失效。
print("-" * 60)
try:
    import numpy as np

    print(f"[OK] numpy       : {np.__version__}")
    if int(np.__version__.split(".")[0]) >= 2:
        print()
        print("[!!] 警告：检测到 NumPy 2.x，但 torch 是针对 NumPy 1.x 编译的。")
        print("     症状：import torch 时刷屏 'A module that was compiled using")
        print("           NumPy 1.x cannot be run in NumPy 2.x'，且 tensor.numpy()、")
        print("           plt.imshow(tensor.numpy()) 等转换会报错。")
        print("     修复：conda install \"numpy<2\" -y   然后重启 kernel")
        print()
except ModuleNotFoundError:
    print("[!!] 缺少 numpy")

# ---------- 3. PyTorch ----------
print("-" * 60)
try:
    import torch
    import torchvision

    print(f"[OK] torch       : {torch.__version__}")
    print(f"[OK] torchvision : {torchvision.__version__}")

    # ---------- 4. 可用设备 ----------
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[OK] CUDA 可用   : {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"          # Apple Silicon GPU
        print("[OK] MPS 可用    : 将使用 Apple Silicon GPU")
    else:
        device = "cpu"
        print("[--] 未检测到 GPU，使用 CPU（MNIST 规模够用，只是慢些）")
    print(f"\n>>> 建议 DEVICE = '{device}'")

    # ---------- 5. 一次真实的前向计算 ----------
    d = torch.device(device)
    x = torch.randn(8, 784, device=d)
    w = torch.randn(784, 10, device=d)
    y = torch.sigmoid(x @ w)
    print(f"[OK] 前向测试通过，输出形状 {tuple(y.shape)}，设备 {y.device}")

    # ---------- 6. torch <-> numpy 转换（画图必需，numpy 冲突就炸在这里）----------
    try:
        arr = y.detach().cpu().numpy()
        back = torch.from_numpy(arr)
        print(f"[OK] tensor <-> numpy 转换正常，形状 {arr.shape}")
        import matplotlib
        print(f"[OK] matplotlib  : {matplotlib.__version__}")
    except Exception as e:
        print(f"[!!] tensor <-> numpy 转换失败: {type(e).__name__}: {e}")
        print('     修复：conda install "numpy<2" -y   然后重启 kernel')
        sys.exit(1)

except ModuleNotFoundError as e:
    print(f"[!!] 缺少依赖: {e}")
    print("\n请在终端运行下面这条命令安装，然后重启 kernel：")
    print("    conda install pytorch torchvision -c pytorch -y")
    sys.exit(1)

# ---------- 5. MNIST 数据 ----------
print("-" * 60)
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.isdir(os.path.join(data_root, "MNIST", "raw")):
    files = os.listdir(os.path.join(data_root, "MNIST", "raw"))
    print(f"[OK] MNIST 数据已存在: {data_root}")
    print(f"     文件数 {len(files)}（已下载过，download=True 不会重复下载）")
else:
    print(f"[--] 未找到数据目录 {data_root}，首次运行会自动下载")

print("=" * 60)
print("自检结束")
