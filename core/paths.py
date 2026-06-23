from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / "assets"
WEIGHTS_ROOT = ASSETS_ROOT / "weights"
CASES_ROOT = ASSETS_ROOT / "cases"
ANNOTATION_ROOT = ASSETS_ROOT / "annotations"
OUTPUT_ROOT = ROOT / "outputs"
UPLOAD_ROOT = ROOT / "uploads"

YOLO_LOCAL_PATH = WEIGHTS_ROOT / "yolo11n.pt"
OSNET_DEFAULT_PATH = WEIGHTS_ROOT / "osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

for path in [OUTPUT_ROOT, UPLOAD_ROOT, WEIGHTS_ROOT, CASES_ROOT, ANNOTATION_ROOT]:
    path.mkdir(parents=True, exist_ok=True)
