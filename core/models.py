from __future__ import annotations

from pathlib import Path
import sys

import torch

from core.paths import YOLO_LOCAL_PATH, OSNET_DEFAULT_PATH


def get_device(prefer_cuda: bool = True, require_cuda: bool = False) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda:0"
    if require_cuda:
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia.")
    return "cpu"


def cuda_status() -> dict:
    return {
        "python_executable": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def load_yolo_model(weight_path: str | Path | None = None):
    from ultralytics import YOLO

    weight = Path(weight_path) if weight_path else YOLO_LOCAL_PATH
    if weight.exists():
        return YOLO(str(weight))
    # fallback: ultralytics akan mencoba download jika internet tersedia
    return YOLO("yolo11n.pt")


def load_osnet_model(weight_path: str | Path | None = None, device: str = "cpu"):
    import torchreid

    weight = Path(weight_path) if weight_path else OSNET_DEFAULT_PATH
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1041,
        loss="softmax",
        pretrained=False,
    )

    if weight.exists():
        checkpoint = torch.load(str(weight), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("classifier"):
                continue
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
    else:
        raise FileNotFoundError(f"Weight OSNet tidak ditemukan: {weight}")

    model = model.to(device)
    model.eval()
    return model
