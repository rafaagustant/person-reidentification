from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_PLACEHOLDER_REASONS = {"", "not merged", "not_merged", "none", "null", "nan", "not applicable", "n/a", "not rejected", "accepted", "rejected", "merged"}


def is_meaningful_reason(value: Any) -> bool:
    return str(value or "").strip().lower().rstrip(".") not in _PLACEHOLDER_REASONS


def format_frame_range(start: Any, end: Any) -> str:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return "Tidak tersedia"
    start_value, end_value = int(start), int(end)
    return str(start_value) if start_value == end_value else f"{start_value}–{end_value}"


def filtered_tracks_view(summary: pd.DataFrame) -> pd.DataFrame:
    filtered = summary[summary.get("is_valid", pd.Series(dtype=bool)) == False].copy() if len(summary) and "is_valid" in summary else pd.DataFrame()
    if len(filtered):
        filtered["filter_reasons"] = filtered.get("filter_reason", "")
        distances = []
        for _, row in filtered.iterrows():
            values = []
            for name, value, limit in [("min_frames", row.get("num_frames"), row.get("min_frames_used")), ("min_crops", row.get("num_crops"), row.get("min_crops_used")), ("min_avg_conf", row.get("avg_conf"), row.get("min_avg_conf_used")), ("min_avg_area", row.get("avg_area"), row.get("min_avg_area_used"))]:
                if name in str(row.get("filter_reason", "")) and pd.notna(value) and pd.notna(limit):
                    values.append(abs(float(limit) - float(value)))
            distances.append(min(values) if values else float("inf"))
        filtered["distance_to_filter_threshold"] = distances
        filtered = filtered.sort_values(["distance_to_filter_threshold", "camera", "track_id"], kind="stable")
    return filtered


def merged_global_gallery_view(global_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group final Global ID members for the merged-track presentation."""
    if not len(global_meta) or not {"global_id", "track_key"}.issubset(global_meta.columns):
        return pd.DataFrame(), pd.DataFrame()

    meta = global_meta.copy()
    if "camera" not in meta:
        meta["camera"] = ""
    for column in ["num_frames", "num_crops", "avg_conf"]:
        if column not in meta:
            meta[column] = pd.NA
    meta["local_track_id"] = meta["track_id"] if "track_id" in meta else pd.NA
    crop_columns = ["representative_crop", "crop_path", "image_path", "sample_path"]
    meta["crop_path"] = ""
    for column in crop_columns:
        if column in meta:
            meta["crop_path"] = meta["crop_path"].mask(meta["crop_path"].eq(""), meta[column].fillna("").astype(str))
    member_columns = [column for column in [
        "global_id", "track_key", "camera", "local_track_id", "crop_path", "num_frames", "num_crops", "avg_conf",
    ] if column in meta]
    members = meta[member_columns].copy()
    members["_track_order"] = pd.to_numeric(members["local_track_id"], errors="coerce").fillna(float("inf"))
    members = members.sort_values(["global_id", "camera", "_track_order", "track_key"], kind="stable").drop(columns="_track_order")

    rows = []
    for global_id, group in members.groupby("global_id", sort=True):
        if len(group) < 2:
            continue
        original = meta[meta["global_id"] == global_id]
        dominant = original["dominant_gt_id"].dropna().mode() if "dominant_gt_id" in original else pd.Series(dtype=object)
        purity = original["global_id_purity"].dropna().iloc[0] if "global_id_purity" in original and original["global_id_purity"].notna().any() else None
        rows.append({
            "global_id": global_id,
            "member_track_keys": ", ".join(group["track_key"].astype(str)),
            "num_member_tracks": int(len(group)),
            "cameras": ", ".join(sorted(group["camera"].dropna().astype(str).unique())),
            "num_cameras": int(group["camera"].nunique()),
            "dominant_gt_id": dominant.iloc[0] if len(dominant) else None,
            "global_id_purity": purity,
        })
    groups = pd.DataFrame(rows)
    if len(groups):
        groups["_global_order"] = pd.to_numeric(groups["global_id"], errors="coerce")
        groups = groups.sort_values(["_global_order", "global_id"], kind="stable").drop(columns="_global_order")
    return groups, members


def global_id_gallery_view(global_meta: pd.DataFrame, gallery_index: pd.DataFrame) -> pd.DataFrame:
    """Prepare one deterministic representative card for every final Global ID."""
    _, members = merged_global_gallery_view(global_meta)
    if not len(members):
        return pd.DataFrame()
    image_by_global_id = {}
    if len(gallery_index) and {"global_id", "image_path"}.issubset(gallery_index.columns):
        image_by_global_id = gallery_index.drop_duplicates("global_id").set_index("global_id")["image_path"].to_dict()

    rows = []
    for global_id, group in members.groupby("global_id", sort=True):
        source = global_meta[global_meta["global_id"] == global_id].copy()
        if "camera" not in source:
            source["camera"] = ""
        for column in ["avg_area", "avg_conf"]:
            if column not in source:
                source[column] = pd.NA
        source["_area"] = pd.to_numeric(source["avg_area"], errors="coerce").fillna(float("-inf"))
        source["_conf"] = pd.to_numeric(source["avg_conf"], errors="coerce").fillna(float("-inf"))
        source["_track_order"] = pd.to_numeric(source.get("track_id", pd.Series(index=source.index, dtype=float)), errors="coerce").fillna(float("inf"))
        representative = source.sort_values(
            ["_area", "_conf", "camera", "_track_order", "track_key"],
            ascending=[False, False, True, True, True], kind="stable",
        ).iloc[0]
        dominant = source["dominant_gt_id"].dropna().mode() if "dominant_gt_id" in source else pd.Series(dtype=object)
        purity = source["global_id_purity"].dropna().iloc[0] if "global_id_purity" in source and source["global_id_purity"].notna().any() else None
        fallback = group[group["track_key"] == representative["track_key"]].iloc[0]
        gallery_path = image_by_global_id.get(global_id)
        if gallery_path is None or pd.isna(gallery_path) or not str(gallery_path).strip():
            gallery_path = fallback["crop_path"]
        rows.append({
            "global_id": global_id,
            "representative_crop_path": gallery_path,
            "representative_track_key": representative["track_key"],
            "representative_camera": representative.get("camera", ""),
            "representative_local_track_id": representative.get("track_id", pd.NA),
            "num_member_tracks": int(len(group)),
            "member_track_keys": ", ".join(group["track_key"].astype(str)),
            "cameras": ", ".join(sorted(group["camera"].dropna().astype(str).unique())),
            "num_cameras": int(group["camera"].nunique()),
            "dominant_gt_id": dominant.iloc[0] if len(dominant) else None,
            "global_id_purity": purity,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["_global_order"] = pd.to_numeric(out["global_id"], errors="coerce")
        out = out.sort_values(["_global_order", "global_id"], kind="stable").drop(columns="_global_order")
    return out


def compact_pairs(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pairs.copy()
    if not len(out):
        return out
    out["threshold"] = threshold
    out["similarity_margin"] = (out["cosine_similarity"] - threshold).abs()
    for suffix in ("a", "b"):
        start_column = f"start_source_frame_{suffix}"
        end_column = f"end_source_frame_{suffix}"
        if start_column not in out:
            out[start_column] = out.get(f"first_frame_{suffix}", pd.NA)
        elif f"first_frame_{suffix}" in out:
            out[start_column] = out[start_column].fillna(out[f"first_frame_{suffix}"])
        if end_column not in out:
            out[end_column] = out.get(f"last_frame_{suffix}", pd.NA)
        elif f"last_frame_{suffix}" in out:
            out[end_column] = out[end_column].fillna(out[f"last_frame_{suffix}"])
        out[f"frame_range_{suffix}"] = [
            format_frame_range(start, end) for start, end in zip(out[start_column], out[end_column])
        ]
    has_source_temporal = out.get("temporal_status", pd.Series(pd.NA, index=out.index)).notna()
    if "overlap_frame_count" not in out:
        out["overlap_frame_count"] = pd.NA
    if "temporal_gap" not in out:
        out["temporal_gap"] = pd.NA
    if "temporal_status" not in out:
        out["temporal_status"] = "Tidak tersedia"
    out.loc[~has_source_temporal, "overlap_frame_count"] = pd.NA
    out.loc[~has_source_temporal, "temporal_gap"] = pd.NA
    out.loc[~has_source_temporal, "temporal_status"] = "Tidak tersedia"
    out = out.drop(columns=["overlap_ratio"], errors="ignore")
    out["association_type"] = out["same_camera"].map({True: "intra_camera", False: "cross_camera"}) if "same_camera" in out else ""
    mnn_labels = {
        "passed": "Diterima",
        "failed": "Ditolak",
        "not_applicable": "Tidak berlaku",
        "not_evaluated": "Tidak dievaluasi",
    }
    if "mnn_status_code" in out:
        out["mnn_status"] = out["mnn_status_code"].map(mnn_labels).fillna("Tidak tersedia")
    elif "is_mnn" in out:
        checked = out["mnn_evaluated"] if "mnn_evaluated" in out else pd.Series(False, index=out.index)
        out["mnn_status"] = pd.Series("Tidak tersedia", index=out.index)
        out.loc[checked & out["is_mnn"].astype(bool), "mnn_status"] = "Diterima"
        out.loc[checked & ~out["is_mnn"].astype(bool), "mnn_status"] = "Ditolak"
    else:
        out["mnn_status"] = "Tidak tersedia"
    merge_status = out["merge_status"] if "merge_status" in out else pd.Series(False, index=out.index)
    out["decision"] = merge_status.map({True: "Diterima", False: "Ditolak"})
    reason = out.get("merge_reason", pd.Series("", index=out.index)).where(~merge_status, "")
    out["rejection_reason"] = reason.where(reason.map(is_meaningful_reason), "")
    return out


def global_id_display_view(meta: pd.DataFrame) -> pd.DataFrame:
    temporal_columns = [
        column for column in meta.columns
        if "source_frame" in column or column in {"source_frames", "first_frame", "last_frame", "temporal_gap", "temporal_overlap"}
    ]
    return meta.drop(columns=temporal_columns, errors="ignore")


def dataframe(result: dict | None, key: str) -> pd.DataFrame:
    value = (result or {}).get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def first_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if len(frame) else {}


def count_tracks(frame: pd.DataFrame) -> int:
    if len(frame) == 0:
        return 0
    if "track_key" in frame:
        return int(frame["track_key"].dropna().nunique())
    if {"camera", "track_id"}.issubset(frame.columns):
        return int(frame[["camera", "track_id"]].drop_duplicates().shape[0])
    return 0


def metric_status(value: Any, *, gt_available: bool = True, applicable: bool = True) -> str:
    if not applicable:
        return "Tidak berlaku"
    if not gt_available:
        return "GT tidak tersedia"
    if value is None or pd.isna(value):
        return "Belum dihitung"
    return ""


def format_count(value: Any, status: str = "") -> str:
    if status:
        return status
    return str(int(value or 0))


def format_ratio(value: Any, status: str = "") -> str:
    if status:
        return status
    return f"{float(value):.3f}" if value is not None and not pd.isna(value) else "Data kosong"


def tracking_view(result: dict | None) -> dict[str, Any]:
    raw = dataframe(result, "raw_tracks_all")
    valid_summary = dataframe(result, "valid_summary_all")
    valid_tracks = dataframe(result, "valid_tracks_all")
    metrics = first_row(dataframe(result, "tracking_standard_metrics_df"))
    if "mota" not in metrics:
        metrics["mota"] = metrics.get("raw_mota_simple", metrics.get("mota_simple"))
    gt_available = bool(metrics.get("score_available", False))
    valid = count_tracks(valid_summary) or count_tracks(valid_tracks)
    standard_by_camera = dataframe(result, "tracking_standard_metrics_by_camera_df")
    if len(standard_by_camera) and "mota" not in standard_by_camera:
        for legacy_key in ("raw_mota_simple", "mota_simple"):
            if legacy_key in standard_by_camera:
                standard_by_camera["mota"] = standard_by_camera[legacy_key]
                break
    return {
        "raw_tracks": count_tracks(raw), "valid_tracks": valid, "raw_tracks_df": raw,
        "valid_summary_df": valid_summary, "valid_tracks_df": valid_tracks, "metrics": metrics,
        "gt_available": gt_available, "camera_status_df": dataframe(result, "camera_status_df"),
        "standard_by_camera_df": standard_by_camera,
        "score_by_camera_df": dataframe(result, "tracking_score_by_camera_df"),
        "local_gallery_df": dataframe(result, "local_gallery_df"),
        "temporal_coverage_df": dataframe(result, "gt_temporal_coverage_df"),
        "temporal_coverage_summary_df": dataframe(result, "gt_temporal_coverage_summary_df"),
        "matched_df": dataframe(result, "matched_df"),
        "filtered_tracks_df": filtered_tracks_view(valid_summary),
    }


def reid_view(result: dict | None) -> dict[str, Any]:
    raw_pairs = dataframe(result, "pair_df")
    metrics = first_row(dataframe(result, "reid_pairwise_eval_df"))
    meta = dataframe(result, "global_meta_df")
    merged_groups, merged_members = merged_global_gallery_view(meta)
    global_cards = global_id_gallery_view(meta, dataframe(result, "global_gallery_df"))
    compact = compact_pairs(raw_pairs, float((result or {}).get("reid_config_used", {}).get("cross_threshold", 0.75)))
    accepted = compact[compact["decision"] == "Diterima"].copy() if len(compact) else pd.DataFrame()
    rejected = compact[compact["decision"] == "Ditolak"].copy() if len(compact) else pd.DataFrame()
    if len(accepted) and {"track_key", "global_id"}.issubset(meta.columns):
        global_mapping = meta.set_index("track_key")["global_id"].to_dict()
        accepted["resulting_global_id"] = accepted["track_a"].map(global_mapping)
    return {
        "global_ids": int(meta["global_id"].nunique()) if "global_id" in meta else 0,
        "merged_pairs": int(raw_pairs["merge_status"].sum()) if "merge_status" in raw_pairs else 0,
        "metrics": metrics, "pairs_df": compact, "raw_pairs_df": raw_pairs, "compact_pairs_df": compact, "global_meta_df": global_id_display_view(meta),
        "gallery_df": dataframe(result, "global_gallery_df"),
        "global_id_gallery_cards_df": global_cards,
        "rendered_paths": (result or {}).get("rendered_paths", []),
        "combined_path": (result or {}).get("combined_path"),
        "merge_gallery_df": dataframe(result, "merge_gallery_df"),
        "merged_global_gallery_df": merged_groups,
        "merged_global_gallery_members_df": merged_members,
        "accepted_pairs_df": accepted,
        "rejected_pairs_df": rejected,
    }


def manifest_metrics(tracking_result: dict | None, reid_result: dict | None) -> dict:
    tracking = tracking_view(tracking_result)
    reid = reid_view(reid_result)
    return {
        "tracking_metrics": tracking["metrics"], "global_id_metrics": reid["metrics"],
        "raw_tracks": tracking["raw_tracks"], "valid_tracks": tracking["valid_tracks"],
        "global_ids": reid["global_ids"], "merged_pairs": reid["merged_pairs"],
    }
