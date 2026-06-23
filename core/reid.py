from __future__ import annotations

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


def build_sampled_track_crop_df(valid_tracks_df: pd.DataFrame, filter_cfg: dict) -> pd.DataFrame:
    if valid_tracks_df is None or len(valid_tracks_df) == 0:
        return pd.DataFrame()

    max_samples = int(filter_cfg.get("max_samples_per_track", 48))
    strategy = filter_cfg.get("crop_selection_strategy", "quality")
    rows = []

    df = valid_tracks_df.copy()
    df["quality_score"] = df["conf"].astype(float) * np.log1p(df["area"].astype(float))

    for track_key, g in df.groupby("track_key"):
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
