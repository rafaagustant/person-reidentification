from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


REID_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_sampled_track_crop_df(
    valid_tracks_df: pd.DataFrame,
    filter_cfg: dict,
    camera_configs: dict | None = None,
) -> pd.DataFrame:
    if valid_tracks_df is None or len(valid_tracks_df) == 0:
        return pd.DataFrame()

    rows = []

    df = valid_tracks_df.copy()
    df["quality_score"] = df["conf"].astype(float) * np.log1p(df["area"].astype(float))

    for track_key, g in df.groupby("track_key"):
        camera = str(g["camera"].iloc[0])
        camera_filter = ((camera_configs or {}).get(camera) or {}).get("filter") or filter_cfg
        max_samples = int(camera_filter.get("max_samples_per_track", 48))
        strategy = str(camera_filter.get("crop_selection_strategy", "quality"))
        if strategy == "uniform":
            g = g.sort_values("frame")
            if len(g) > max_samples:
                idx = np.linspace(0, len(g) - 1, max_samples).round().astype(int)
                chosen = g.iloc[idx]
            else:
                chosen = g
        else:
            chosen = g.sort_values("quality_score", ascending=False).head(max_samples).sort_values("frame")
        rows.append(chosen)

    sampled = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return sampled.drop(columns=["quality_score"], errors="ignore")


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_embedding_manifest(
    valid_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    osnet_weight: str | Path | None,
    feature_dim: int | None = None,
) -> dict:
    weight_path = Path(osnet_weight) if osnet_weight else None
    weight_fingerprint = None
    if weight_path and weight_path.exists():
        digest = hashlib.sha256()
        with weight_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        weight_fingerprint = digest.hexdigest()
    selected = sampled_df.copy() if sampled_df is not None else pd.DataFrame()
    selected_columns = [col for col in ["track_key", "camera", "frame", "crop_path"] if col in selected]
    valid_columns = [col for col in ["track_key", "camera", "frame", "crop_path"] if col in valid_df]
    return {
        "cache_format_version": 1,
        "model_architecture": "osnet_x1_0",
        "weight_path": str(weight_path) if weight_path else None,
        "weight_fingerprint": weight_fingerprint,
        "embedding_dim": int(feature_dim) if feature_dim is not None else None,
        "valid_track_fingerprint": _fingerprint(valid_df[valid_columns].to_dict("records") if valid_columns else []),
        "selected_crop_fingerprint": _fingerprint(selected[selected_columns].to_dict("records") if selected_columns else []),
        "max_samples_per_track": sorted(set(selected.groupby("track_key").size().tolist())) if len(selected) else [],
        "crop_selection_strategy": "per_camera_config",
        "preprocessing": {"resize": [256, 128], "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225]},
    }


def embedding_cache_is_valid(manifest_path: str | Path, expected_manifest: dict, features_path: str | Path) -> bool:
    try:
        actual = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        features = np.load(features_path, mmap_mode="r")
    except Exception:
        return False
    expected = dict(expected_manifest)
    expected["embedding_dim"] = int(features.shape[1]) if features.ndim == 2 else None
    return actual == expected


@torch.no_grad()
def extract_embeddings_from_sampled_df(model, sampled_df: pd.DataFrame, use_cuda: bool = True, batch_size: int = 32, progress_callback=None):
    if sampled_df is None or len(sampled_df) == 0:
        return sampled_df.copy(), np.empty((0, 0), dtype=np.float32)

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rows = []
    feats = []
    batch_imgs = []
    batch_rows = []
    processed = 0
    total = len(sampled_df)

    def flush_batch():
        nonlocal batch_imgs, batch_rows, processed
        if not batch_imgs:
            return
        batch = torch.stack(batch_imgs).to(device)
        out = model(batch)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        out_np = out.detach().cpu().numpy().astype(np.float32)
        for r, f in zip(batch_rows, out_np):
            rows.append(r)
            feats.append(f)
        processed += len(batch_rows)
        if progress_callback is not None:
            progress_callback(processed, total)
        batch_imgs = []
        batch_rows = []

    for _, row in sampled_df.iterrows():
        crop_path = Path(row["crop_path"])
        if not crop_path.exists():
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total)
            continue
        try:
            img = Image.open(crop_path).convert("RGB")
            batch_imgs.append(REID_TRANSFORM(img))
            batch_rows.append(row.to_dict())
        except Exception:
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total)
            continue
        if len(batch_imgs) >= batch_size:
            flush_batch()
    flush_batch()

    out_df = pd.DataFrame(rows)
    feat_arr = np.vstack(feats).astype(np.float32) if feats else np.empty((0, 0), dtype=np.float32)
    return out_df, feat_arr


def build_track_embedding_df(crop_rows_df: pd.DataFrame, crop_features: np.ndarray):
    if crop_rows_df is None or len(crop_rows_df) == 0 or crop_features.size == 0:
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32)

    rows = []
    feats = []
    crop_rows_df = crop_rows_df.reset_index(drop=True)

    for track_key, g in crop_rows_df.groupby("track_key"):
        idx = g.index.to_numpy()
        feat = crop_features[idx].mean(axis=0)
        norm = np.linalg.norm(feat) + 1e-12
        feat = (feat / norm).astype(np.float32)
        feats.append(feat)
        rep = g.assign(q=g["conf"].astype(float) * np.log1p(g["area"].astype(float))).sort_values("q", ascending=False).iloc[0]
        rows.append({
            "track_key": track_key,
            "camera": g["camera"].iloc[0],
            "track_id": int(g["track_id"].iloc[0]),
            "num_crops": int(len(g)),
            "first_frame": int(g["frame"].min()),
            "last_frame": int(g["frame"].max()),
            "avg_conf": float(g["conf"].mean()),
            "avg_area": float(g["area"].mean()),
            "representative_crop": rep.get("crop_path", ""),
        })

    track_df = pd.DataFrame(rows).reset_index(drop=True)
    track_feat = np.vstack(feats).astype(np.float32)
    return track_df, track_feat
