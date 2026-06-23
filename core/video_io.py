from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def get_video_info(video_path: str | Path) -> dict:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"exists": video_path.exists(), "error": "Video tidak dapat dibuka."}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "exists": True,
        "path": str(video_path),
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": frames / fps if fps else 0,
    }


def read_frame(video_path: str | Path, frame_idx: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def make_preview_grid(video_files: dict, frame_idx: int = 0, cell_w: int = 420):
    frames = []
    for camera, path in video_files.items():
        frame = read_frame(path, frame_idx)
        if frame is None:
            frame = np.zeros((240, cell_w, 3), dtype=np.uint8)
            cv2.putText(frame, f"{camera}: video tidak tersedia", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            h, w = frame.shape[:2]
            scale = cell_w / max(1, w)
            frame = cv2.resize(frame, (cell_w, int(h * scale)))
            cv2.putText(frame, camera, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        frames.append(frame)
    if not frames:
        return None
    max_h = max(f.shape[0] for f in frames)
    padded = []
    for f in frames:
        if f.shape[0] < max_h:
            pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        padded.append(f)
    return np.hstack(padded)
