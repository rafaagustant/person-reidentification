from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml

from utils.helpers import ensure_dir


TRACK_COLUMNS = [
    "camera",
    "frame",
    "track_id",
    "track_key",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "width",
    "height",
    "area",
    "crop_path",
]


SUMMARY_COLUMNS = [
    "camera",
    "track_id",
    "track_key",
    "num_frames",
    "num_crops",
    "avg_conf",
    "avg_area",
    "first_frame",
    "last_frame",
    "representative_crop",
    "min_frames_used",
    "min_crops_used",
    "min_avg_conf_used",
    "min_avg_area_used",
    "pass_min_frames",
    "pass_min_crops",
    "pass_min_avg_conf",
    "pass_min_avg_area",
    "is_valid",
    "status",
    "filter_reason",
]


def make_botsort_yaml(config: dict, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)

    tracker_cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": float(config.get("track_high_thresh", 0.25)),
        "track_low_thresh": float(config.get("track_low_thresh", 0.10)),
        "new_track_thresh": float(config.get("new_track_thresh", 0.25)),
        "track_buffer": int(config.get("track_buffer", 30)),
        "match_thresh": float(config.get("match_thresh", 0.80)),
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.50,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "model": "auto",
    }

    path = output_dir / "generated_tracker_config_used.yaml"
    header = "\n".join(
        [
            "# Generated tracker config used by Ultralytics model.track().",
            "# Detection args are passed to model.track(), not parsed as BoT-SORT YAML fields.",
            f"# yolo_conf: {float(config.get('yolo_conf', 0.25))}",
            f"# yolo_iou: {float(config.get('yolo_iou', 0.50))}",
            f"# imgsz: {int(config.get('imgsz', 640))}",
            "",
        ]
    )
    path.write_text(header + yaml.safe_dump(tracker_cfg, sort_keys=False), encoding="utf-8")
    return path


def save_tracking_runtime_manifest(
    config: dict,
    tracker_yaml: str | Path,
    output_dir: str | Path,
    tracking_fingerprint: str | None = None,
    device=None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    tracker_yaml = Path(tracker_yaml)
    manifest = {
        "tracking_fingerprint": tracking_fingerprint,
        "tracker_yaml_path": str(tracker_yaml),
        "model_track_args": {
            "conf": float(config.get("yolo_conf", 0.25)),
            "iou": float(config.get("yolo_iou", 0.50)),
            "imgsz": int(config.get("imgsz", 640)),
            "classes": [0],
            "persist": True,
            "device": device,
        },
        "tracking_config": {
            "yolo_conf": float(config.get("yolo_conf", 0.25)),
            "yolo_iou": float(config.get("yolo_iou", 0.50)),
            "imgsz": int(config.get("imgsz", 640)),
            "track_high_thresh": float(config.get("track_high_thresh", 0.25)),
            "track_low_thresh": float(config.get("track_low_thresh", 0.10)),
            "new_track_thresh": float(config.get("new_track_thresh", 0.25)),
            "match_thresh": float(config.get("match_thresh", 0.80)),
            "track_buffer": int(config.get("track_buffer", 30)),
        },
    }
    path = output_dir / "tracking_runtime_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def reset_yolo_tracker(model) -> None:
    try:
        if hasattr(model, "predictor"):
            model.predictor = None
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def empty_track_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACK_COLUMNS)


def safe_progress(progress_callback, camera: str, done: int, total: int) -> None:
    if progress_callback is None:
        return

    try:
        progress_callback(camera, done, total)
    except Exception:
        pass


def clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def run_track_once(model, frame, tracker_yaml: Path, config: dict, device):
    return model.track(
        frame,
        persist=True,
        tracker=str(tracker_yaml),
        classes=[0],
        conf=float(config.get("yolo_conf", 0.25)),
        iou=float(config.get("yolo_iou", 0.50)),
        imgsz=int(config.get("imgsz", 640)),
        device=device,
        verbose=False,
    )


def run_track_with_retry(model, frame, tracker_yaml: Path, config: dict, device):
    try:
        return run_track_once(
            model=model,
            frame=frame,
            tracker_yaml=tracker_yaml,
            config=config,
            device=device,
        )

    except TypeError as e:
        message = str(e)

        if "NoneType" in message and "subscriptable" in message:
            reset_yolo_tracker(model)
            return run_track_once(
                model=model,
                frame=frame,
                tracker_yaml=tracker_yaml,
                config=config,
                device=device,
            )

        raise


def process_video_tracking(
    model,
    video_path: str | Path,
    camera: str,
    output_dir: str | Path,
    config: dict,
    use_cuda: bool = True,
    progress_callback=None,
    tracking_fingerprint: str | None = None,
) -> pd.DataFrame:
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)
    crop_root = ensure_dir(output_dir / "crops")
    tracker_yaml = make_botsort_yaml(config, output_dir)
    device = 0 if use_cuda and torch.cuda.is_available() else "cpu"
    save_tracking_runtime_manifest(config, tracker_yaml, output_dir, tracking_fingerprint, device=device)

    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows = []
    frame_idx = 0

    reset_yolo_tracker(model)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            results = run_track_with_retry(
                model=model,
                frame=frame,
                tracker_yaml=tracker_yaml,
                config=config,
                device=device,
            )

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.detach().cpu().numpy()
                tids = results[0].boxes.id.detach().cpu().numpy().astype(int)
                confs = results[0].boxes.conf.detach().cpu().numpy()

                height, width = frame.shape[:2]

                for box, tid, conf in zip(boxes, tids, confs):
                    clipped = clip_box(box, width=width, height=height)

                    if clipped is None:
                        continue

                    x1, y1, x2, y2 = clipped
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    track_key = f"{camera}_T{int(tid)}"
                    track_dir = ensure_dir(crop_root / f"track_{int(tid)}")
                    crop_path = track_dir / f"frame_{frame_idx:06d}.jpg"

                    cv2.imwrite(str(crop_path), crop)

                    rows.append(
                        {
                            "camera": camera,
                            "frame": int(frame_idx),
                            "track_id": int(tid),
                            "track_key": track_key,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "conf": float(conf),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                            "area": int((x2 - x1) * (y2 - y1)),
                            "crop_path": str(crop_path),
                        }
                    )

            frame_idx += 1
            safe_progress(progress_callback, camera, frame_idx, total_frames)

    finally:
        cap.release()
        reset_yolo_tracker(model)

    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    df.to_csv(output_dir / "local_tracks.csv", index=False)
    return df


def summarize_tracks(track_df: pd.DataFrame) -> pd.DataFrame:
    if track_df is None or len(track_df) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []

    for track_key, group in track_df.groupby("track_key"):
        rep = group.sort_values(["area", "conf"], ascending=False).iloc[0]
        num_crops = int(group["crop_path"].fillna("").astype(str).ne("").sum()) if "crop_path" in group else 0

        rows.append(
            {
                "camera": group["camera"].iloc[0],
                "track_id": int(group["track_id"].iloc[0]),
                "track_key": track_key,
                "num_frames": int(group["frame"].nunique()),
                "num_crops": num_crops,
                "avg_conf": float(group["conf"].mean()),
                "avg_area": float(group["area"].mean()),
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "representative_crop": rep.get("crop_path", ""),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["camera", "track_id"])
        .reset_index(drop=True)
    )


def filter_valid_tracks(track_df: pd.DataFrame, filter_cfg: dict):
    if track_df is None:
        track_df = empty_track_df()

    summary = summarize_tracks(track_df)

    if len(summary) == 0:
        return track_df.copy(), summary

    min_frames = int(filter_cfg.get("min_frames", filter_cfg.get("min_crops", 1)))
    min_crops = int(filter_cfg.get("min_crops", 1))
    min_avg_conf = float(filter_cfg.get("min_avg_conf", 0.0))
    min_avg_area = float(filter_cfg.get("min_avg_area", 0.0))

    summary["min_frames_used"] = min_frames
    summary["min_crops_used"] = min_crops
    summary["min_avg_conf_used"] = min_avg_conf
    summary["min_avg_area_used"] = min_avg_area
    summary["pass_min_frames"] = summary["num_frames"] >= min_frames
    summary["pass_min_crops"] = summary["num_crops"] >= min_crops
    summary["pass_min_avg_conf"] = summary["avg_conf"] >= min_avg_conf
    summary["pass_min_avg_area"] = summary["avg_area"] >= min_avg_area
    summary["is_valid"] = (
        summary["pass_min_frames"]
        & summary["pass_min_crops"]
        & summary["pass_min_avg_conf"]
        & summary["pass_min_avg_area"]
    )
    summary["status"] = summary["is_valid"].map(lambda ok: "valid" if ok else "filtered")

    reason_columns = [
        ("pass_min_frames", "min_frames"),
        ("pass_min_crops", "min_crops"),
        ("pass_min_avg_conf", "min_avg_conf"),
        ("pass_min_avg_area", "min_avg_area"),
    ]
    summary["filter_reason"] = summary.apply(
        lambda row: ""
        if bool(row["is_valid"])
        else ", ".join(reason for col, reason in reason_columns if not bool(row[col])),
        axis=1,
    )

    keep = summary[summary["is_valid"]].copy()

    keep_keys = set(keep["track_key"].tolist())
    valid_df = track_df[track_df["track_key"].isin(keep_keys)].copy()

    return valid_df.reset_index(drop=True), summary.reset_index(drop=True)
