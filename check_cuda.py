from __future__ import annotations

import sys

import torch


def main() -> int:
    cuda_available = torch.cuda.is_available()

    print(f"sys.executable: {sys.executable}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if cuda_available:
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        print("CUDA OK")
        return 0

    if "+cpu" in str(torch.__version__).lower() or torch.version.cuda is None:
        print("CUDA NOT AVAILABLE: PyTorch CPU-only build is installed.")
    else:
        print("CUDA NOT AVAILABLE: CUDA build is installed, but no CUDA device is available.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
