from __future__ import annotations

from pathlib import Path
import json
import shutil
import pandas as pd
from PIL import Image

from utils.helpers import ensure_dir


GALLERY_TILE_SIZE = (224, 336)
GALLERY_BG = (18, 20, 24)


def make_crop_tile(
    image_path: str | Path,
    output_path: str | Path,
    size: tuple[int, int] = GALLERY_TILE_SIZE,
    bg_color: tuple[int, int, int] = GALLERY_BG,
) -> Path | None:
    src = Path(image_path)
    if not src.exists():
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        canvas = Image.new("RGB", size, bg_color)
        scale = min(size[0] / max(1, img.width), size[1] / max(1, img.height))
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        x = (size[0] - new_size[0]) // 2
        y = (size[1] - new_size[1]) // 2
        canvas.paste(resized, (x, y))
        canvas.save(output_path, quality=92)
    return output_path

def make_merge_pair_image(
    crop_a: str | Path,
    crop_b: str | Path,
    output_path: str | Path,
    tile_size: tuple[int, int] = GALLERY_TILE_SIZE,
) -> Path | None:
    temp_dir = Path(output_path).parent / "_tiles_tmp"
    tile_a = make_crop_tile(crop_a, temp_dir / "a.jpg", size=tile_size)
    tile_b = make_crop_tile(crop_b, temp_dir / "b.jpg", size=tile_size)
    if tile_a is None or tile_b is None:
        return None

    gap = 8
    w = tile_size[0] * 2 + gap
    h = tile_size[1]
    canvas = Image.new("RGB", (w, h), GALLERY_BG)
    with Image.open(tile_a) as a, Image.open(tile_b) as b:
        canvas.paste(a.convert("RGB"), (0, 0))
        canvas.paste(b.convert("RGB"), (tile_size[0] + gap, 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_path


def build_local_track_gallery(
    valid_summary_df: pd.DataFrame,
    output_dir: str | Path,
    extra_meta: dict | None = None,
) -> pd.DataFrame:
    gallery_dir = ensure_dir(output_dir)
    rows = []
    if valid_summary_df is None or len(valid_summary_df) == 0:
        return pd.DataFrame()

    for _, r in valid_summary_df.iterrows():
        src = Path(str(r.get("representative_crop", "")))
        if not src.exists():
            continue
        name = f"{r['track_key']}.jpg"
        dst = gallery_dir / name
        if make_crop_tile(src, dst) is None:
            continue
        row = {
            "track_key": r["track_key"],
            "camera": r["camera"],
            "track_id": int(r["track_id"]),
            "image_path": str(dst),
            "num_frames": int(r.get("num_frames", 0)),
            "avg_conf": float(r.get("avg_conf", 0.0)),
            "avg_area": float(r.get("avg_area", 0.0)),
        }
        if "dominant_gt_id" in r:
            row["dominant_gt_id"] = r.get("dominant_gt_id")
        if "track_purity" in r:
            row["track_purity"] = r.get("track_purity")
        if "track_gt_purity" in r:
            row["track_purity"] = r.get("track_gt_purity")
        if "profile" in r:
            row["profile"] = r.get("profile")
        if "config_mode" in r:
            row["config_mode"] = r.get("config_mode")
        if extra_meta:
            row.update(extra_meta)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["camera", "track_id", "track_key"], kind="stable")
    out.to_csv(gallery_dir / "local_gallery_index.csv", index=False)
    return out


def build_global_id_gallery(global_meta_df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    gallery_dir = ensure_dir(output_dir)
    rows = []
    if global_meta_df is None or len(global_meta_df) == 0:
        return pd.DataFrame()

    for gid, g in global_meta_df.groupby("global_id"):
        g = g.sort_values(["camera", "track_id"], kind="stable")
        rep = g.sort_values(["avg_area", "avg_conf"], ascending=False).iloc[0]
        src = Path(str(rep.get("representative_crop", "")))
        if not src.exists():
            continue
        dst = gallery_dir / f"GID_{int(gid):03d}.jpg"
        if make_crop_tile(src, dst) is None:
            continue
        members = []
        for _, member in g.iterrows():
            member_src = Path(str(member.get("representative_crop", "")))
            thumb_path = gallery_dir / f"GID_{int(gid):03d}_{member['track_key']}.jpg"
            thumb = make_crop_tile(member_src, thumb_path, size=(112, 168)) if member_src.exists() else None
            members.append({
                "camera": member.get("camera"),
                "track_id": int(member.get("track_id")) if pd.notna(member.get("track_id")) else None,
                "track_key": member.get("track_key"),
                "num_frames": int(member.get("num_frames", 0) or 0),
                "avg_conf": float(member.get("avg_conf", 0.0) or 0.0),
                "dominant_gt_id": member.get("dominant_gt_id") if "dominant_gt_id" in member else None,
                "track_purity": member.get("track_purity") if "track_purity" in member else None,
                "thumbnail_path": str(thumb) if thumb else "",
            })
        rows.append({
            "global_id": int(gid),
            "tracks": ", ".join(g["track_key"].astype(str).tolist()),
            "cameras": ", ".join(sorted(set(g["camera"].astype(str).tolist()))),
            "image_path": str(dst),
            "num_tracks": int(g["track_key"].nunique()),
            "members_json": json.dumps(members, ensure_ascii=False),
        })
    out = pd.DataFrame(rows).sort_values("global_id") if rows else pd.DataFrame()
    out.to_csv(gallery_dir / "gallery_index.csv", index=False)
    return out


def build_merge_gallery(
    merged_pairs_df: pd.DataFrame,
    global_meta_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    gallery_dir = ensure_dir(output_dir)
    rows = []
    if merged_pairs_df is None or len(merged_pairs_df) == 0:
        out = pd.DataFrame()
        out.to_csv(gallery_dir / "merge_gallery_index.csv", index=False)
        return out
    if global_meta_df is None or len(global_meta_df) == 0 or "track_key" not in global_meta_df:
        out = pd.DataFrame()
        out.to_csv(gallery_dir / "merge_gallery_index.csv", index=False)
        return out

    meta = global_meta_df.set_index("track_key").to_dict("index")
    merged = merged_pairs_df.copy()
    sort_cols = [col for col in ["global_id", "cosine_similarity"] if col in merged.columns]
    if sort_cols:
        ascending = [True, False][:len(sort_cols)]
        merged = merged.sort_values(sort_cols, ascending=ascending, kind="stable")

    for idx, pair in merged.reset_index(drop=True).iterrows():
        track_a = str(pair.get("track_a", ""))
        track_b = str(pair.get("track_b", ""))
        info_a = meta.get(track_a, {})
        info_b = meta.get(track_b, {})
        crop_a = info_a.get("representative_crop")
        crop_b = info_b.get("representative_crop")
        if not crop_a or not crop_b:
            continue
        camera_a = pair.get("camera_a", info_a.get("camera"))
        camera_b = pair.get("camera_b", info_b.get("camera"))
        track_id_a = pair.get("track_id_a", info_a.get("track_id"))
        track_id_b = pair.get("track_id_b", info_b.get("track_id"))
        global_id = info_a.get("global_id", info_b.get("global_id"))
        merge_type = "intra_camera" if str(camera_a) == str(camera_b) else "cross_camera"
        similarity = pair.get("cosine_similarity", pair.get("similarity", None))
        similarity_text = "-" if pd.isna(similarity) else f"{float(similarity):.3f}"
        metadata = {
            "global_id": int(global_id) if pd.notna(global_id) else None,
            "merge_type": merge_type,
            "camera_a": camera_a,
            "track_id_a": int(track_id_a) if pd.notna(track_id_a) else None,
            "camera_b": camera_b,
            "track_id_b": int(track_id_b) if pd.notna(track_id_b) else None,
            "similarity": similarity_text,
            "merged": bool(pair.get("merge_status", True)),
        }
        image_name = f"merge_{idx + 1:03d}_{track_a}_{track_b}.jpg".replace("/", "_").replace("\\", "_")
        image_path = make_merge_pair_image(
            crop_a,
            crop_b,
            gallery_dir / image_name,
        )
        if image_path is None:
            continue
        rows.append({
            "image_path": str(image_path),
            "track_a": track_a,
            "track_b": track_b,
            "global_id": metadata["global_id"],
            "merge_type": merge_type,
            "camera_a": camera_a,
            "track_id_a": metadata["track_id_a"],
            "camera_b": camera_b,
            "track_id_b": metadata["track_id_b"],
            "similarity": None if pd.isna(similarity) else float(similarity),
            "merged": metadata["merged"],
            "merge_reason": pair.get("merge_reason", ""),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["global_id", "similarity"], ascending=[True, False], kind="stable")
    out.to_csv(gallery_dir / "merge_gallery_index.csv", index=False)
    return out
