from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _first_row(df: pd.DataFrame | None) -> dict:
    if df is None or len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "valid"])


def build_tracking_diagnostics(
    raw_tracks_df: pd.DataFrame | None,
    valid_tracks_df: pd.DataFrame | None,
    valid_summary_df: pd.DataFrame | None,
    tracking_eval_df: pd.DataFrame | None = None,
    gt_coverage_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw = raw_tracks_df if raw_tracks_df is not None else pd.DataFrame()
    valid = valid_tracks_df if valid_tracks_df is not None else pd.DataFrame()
    summary = valid_summary_df if valid_summary_df is not None else pd.DataFrame()
    eval_row = _first_row(tracking_eval_df)

    per_camera = {}
    valid_summary = summary
    if len(summary) and "is_valid" in summary.columns:
        valid_summary = summary[_truthy_series(summary["is_valid"])].copy()
    if len(summary) and "camera" in summary.columns:
        per_camera = valid_summary.groupby("camera")["track_key"].nunique().to_dict()

    row = {
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "num_valid_tracks": int(valid_summary["track_key"].nunique()) if len(valid_summary) and "track_key" in valid_summary else 0,
        "num_valid_tracks_per_camera": json.dumps(per_camera, ensure_ascii=False),
        "avg_conf_mean": _safe_float(valid_summary["avg_conf"].mean()) if len(valid_summary) and "avg_conf" in valid_summary else None,
        "avg_area_mean": _safe_float(valid_summary["avg_area"].mean()) if len(valid_summary) and "avg_area" in valid_summary else None,
        "first_frame_min": _safe_int(valid["frame"].min(), None) if len(valid) and "frame" in valid else None,
        "last_frame_max": _safe_int(valid["frame"].max(), None) if len(valid) and "frame" in valid else None,
        "gt_available": bool(tracking_eval_df is not None and len(tracking_eval_df)),
        "precision": _safe_float(eval_row.get("pred_match_rate")),
        "recall": _safe_float(eval_row.get("gt_coverage_rate")),
        "f1": None,
        "gt_coverage_rate": _safe_float(eval_row.get("gt_coverage_rate")),
        "false_positive_rate": _safe_float(eval_row.get("false_positive_rate")),
        "fragmentation_avg": _safe_float(eval_row.get("fragmentation_avg")),
        "mean_track_purity": _safe_float(eval_row.get("mean_track_purity")),
        "id_switch_count": _safe_int(eval_row.get("id_switch_count"), None),
    }

    if row["precision"] is not None and row["recall"] is not None:
        denom = row["precision"] + row["recall"]
        row["f1"] = (2 * row["precision"] * row["recall"] / denom) if denom else 0.0

    if gt_coverage_df is not None and len(gt_coverage_df) and "coverage_per_gt" in gt_coverage_df:
        row["gt_coverage_rate"] = _safe_float(gt_coverage_df["coverage_per_gt"].mean(), row["gt_coverage_rate"])

    return pd.DataFrame([row])


def build_tracking_tuning_hints(result: dict) -> list[dict]:
    diag_df = result.get("tracking_diagnostics_df")
    row = _first_row(diag_df)
    hints = []

    recall = _safe_float(row.get("recall"))
    fp_rate = _safe_float(row.get("false_positive_rate"))
    frag = _safe_float(row.get("fragmentation_avg"))
    purity = _safe_float(row.get("mean_track_purity"))
    avg_area = _safe_float(row.get("avg_area_mean"))
    raw_rows = _safe_int(row.get("raw_rows"))
    valid_rows = _safe_int(row.get("valid_rows"))

    if recall is not None and recall < 0.60:
        hints.append({"stage": "tracking", "hint": "Recall rendah. Coba turunkan yolo_conf, naikkan imgsz, atau longgarkan filter track."})
    if fp_rate is not None and fp_rate > 0.35:
        hints.append({"stage": "tracking", "hint": "False positive tinggi. Coba naikkan yolo_conf, min_avg_conf, atau min_avg_area."})
    if purity is not None and purity < 0.85:
        hints.append({"stage": "tracking", "hint": "Mixed track tinggi atau track purity rendah. Coba naikkan match_thresh atau pendekkan track_buffer."})
    if frag is not None and frag > 1.5:
        hints.append({"stage": "tracking", "hint": "Fragmentasi tinggi. Coba naikkan track_buffer atau aktifkan intra-camera Re-ID."})
    if avg_area is not None and avg_area < 1500:
        hints.append({"stage": "tracking", "hint": "Rata-rata area kecil. Tinjau min_avg_area agar track kecil yang valid tidak ikut terbuang."})
    if raw_rows and valid_rows == 0:
        hints.append({"stage": "tracking", "hint": "Tidak ada track valid. Filter kemungkinan terlalu ketat untuk video ini."})
    if not hints:
        hints.append({"stage": "tracking", "hint": "Tidak ada masalah besar dari ringkasan saat ini."})
    return hints


def build_reid_diagnostics(
    track_embedding_df: pd.DataFrame | None,
    pair_df: pd.DataFrame | None,
    global_meta_df: pd.DataFrame | None,
    global_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tracks = track_embedding_df if track_embedding_df is not None else pd.DataFrame()
    pairs = pair_df if pair_df is not None else pd.DataFrame()
    meta = global_meta_df if global_meta_df is not None else pd.DataFrame()
    summary_row = _first_row(global_summary_df)

    merged = pairs[pairs["merge_status"] == True] if len(pairs) and "merge_status" in pairs else pd.DataFrame()
    cross = pairs[pairs["same_camera"] == False] if len(pairs) and "same_camera" in pairs else pd.DataFrame()
    cross_merged = merged[merged["same_camera"] == False] if len(merged) and "same_camera" in merged else pd.DataFrame()
    intra_merged = merged[merged["same_camera"] == True] if len(merged) and "same_camera" in merged else pd.DataFrame()
    unmerged_cross = cross[cross["merge_status"] == False] if len(cross) and "merge_status" in cross else pd.DataFrame()
    reason_counts = merged["merge_reason"].value_counts().to_dict() if len(merged) and "merge_reason" in merged else {}

    row = {
        "num_local_tracks": int(tracks["track_key"].nunique()) if len(tracks) and "track_key" in tracks else 0,
        "num_global_ids": int(meta["global_id"].nunique()) if len(meta) and "global_id" in meta else 0,
        "num_pairs": int(len(pairs)),
        "num_merged_pairs": int(len(merged)),
        "num_cross_camera_pairs": int(len(cross)),
        "num_cross_camera_merged": int(len(cross_merged)),
        "num_intra_camera_merged": int(len(intra_merged)),
        "max_cross_similarity": _safe_float(cross["cosine_similarity"].max()) if len(cross) else None,
        "top_unmerged_cross_similarity": _safe_float(unmerged_cross["cosine_similarity"].max()) if len(unmerged_cross) else None,
        "merge_reason_counts": json.dumps(reason_counts, ensure_ascii=False),
        "identity_precision": _safe_float(summary_row.get("pure_global_id_rate")),
        "identity_recall": None,
        "identity_f1": None,
    }

    if row["identity_precision"] is not None:
        row["identity_recall"] = row["identity_precision"]
        row["identity_f1"] = row["identity_precision"]

    return pd.DataFrame([row])


def build_reid_tuning_hints(result: dict, config: dict | None = None) -> list[dict]:
    diag_df = result.get("reid_diagnostics_df")
    row = _first_row(diag_df)
    reid_cfg = (config or {}).get("reid", config or {})
    hints = []

    cross_threshold = _safe_float(reid_cfg.get("cross_threshold"), 0.75)
    cross_merged = _safe_int(row.get("num_cross_camera_merged"))
    intra_merged = _safe_int(row.get("num_intra_camera_merged"))
    max_cross = _safe_float(row.get("max_cross_similarity"))
    top_unmerged = _safe_float(row.get("top_unmerged_cross_similarity"))
    merged_pairs = _safe_int(row.get("num_merged_pairs"))
    identity_precision = _safe_float(row.get("identity_precision"))

    if cross_merged == 0 and max_cross is not None and abs(max_cross - cross_threshold) <= 0.04:
        hints.append({"stage": "reid", "hint": "Belum ada merge lintas kamera, tetapi similarity tertinggi dekat threshold. Turunkan cross_threshold sedikit jika GT mendukung."})
    if cross_merged > 3:
        hints.append({"stage": "reid", "hint": "Merge lintas kamera cukup banyak. Naikkan cross_threshold atau aktifkan MNN untuk menekan false merge."})
    if intra_merged == 0 and merged_pairs == 0:
        hints.append({"stage": "reid", "hint": "Tidak ada fragment yang tersambung. Jika track satu kamera terpecah, aktifkan strict intra atau turunkan intra_threshold."})
    if identity_precision is not None and identity_precision < 0.90:
        hints.append({"stage": "reid", "hint": "Evaluasi GT menunjukkan indikasi false merge. Naikkan threshold dan perketat filter track."})
    if top_unmerged is not None and cross_threshold is not None and top_unmerged >= cross_threshold:
        hints.append({"stage": "reid", "hint": "Ada pasangan similarity tinggi yang tidak di-merge. Cek MNN dan overlap temporal sebelum menurunkan threshold."})
    if not hints:
        hints.append({"stage": "reid", "hint": "Tidak ada masalah besar dari ringkasan saat ini."})
    return hints


def save_tuning_hints(path: str | Path, hints: list[dict]) -> None:
    Path(path).write_text(json.dumps(hints, ensure_ascii=False, indent=2), encoding="utf-8")
