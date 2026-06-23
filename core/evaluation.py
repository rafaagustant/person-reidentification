from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from core.paths import ROOT
except Exception:
    ROOT = Path(".").resolve()

try:
    from config.gt_cases import with_gt_meta
except Exception:
    def with_gt_meta(case: dict) -> dict:
        return dict(case)

FRAME_KEYS = ["source_frame", "frame", "frame_id", "frame_index", "frame_number", "frameNumber", "image_id"]
ID_KEYS = ["person_id", "identity_id", "identity", "gt_id", "track_id", "subject_id", "object_id", "id"]
BBOX_KEYS = ["BboxP", "bboxP", "bbox", "box", "bounding_box", "rect"]
LIST_KEYS = ["frames", "annotations", "objects", "people", "persons", "detections", "labels", "items"]
EMPTY_GT_COLUMNS = ["camera", "source_frame", "gt_id", "x1", "y1", "x2", "y2", "gt_area"]


def resolve_path(path_value) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _case(case: dict) -> dict:
    return with_gt_meta(case)


def add_source_frame(pred_df: pd.DataFrame, case: dict) -> pd.DataFrame:
    c = _case(case)
    pred_df = pred_df.copy()
    if len(pred_df) == 0:
        pred_df["source_frame"] = []
        return pred_df
    if "source_frame_start" not in c:
        pred_df["source_frame"] = pred_df["frame"].astype(int)
        return pred_df
    local_frame_start = int(c.get("local_frame_start", 0))
    source_frame_start = int(c["source_frame_start"])
    pred_df["source_frame"] = source_frame_start + pred_df["frame"].astype(int) - local_frame_start
    return pred_df


def find_annotation_file(case: dict, camera: str) -> Path | None:
    c = _case(case)
    ann_dir = resolve_path(c.get("annotation_dir", ""))
    if not ann_dir.exists():
        return None
    candidates = sorted(ann_dir.glob(f"*{camera}*.json"))
    return candidates[0] if candidates else None


def _get_first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_bbox(value, bbox_format="xyxy"):
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ["x", "y", "w", "h"]):
            x, y, w, h = value["x"], value["y"], value["w"], value["h"]
            return float(x), float(y), float(x) + float(w), float(y) + float(h)
        if all(k in value for k in ["x", "y", "width", "height"]):
            x, y, w, h = value["x"], value["y"], value["width"], value["height"]
            return float(x), float(y), float(x) + float(w), float(y) + float(h)
        if all(k in value for k in ["left", "top", "right", "bottom"]):
            return float(value["left"]), float(value["top"]), float(value["right"]), float(value["bottom"])
        if all(k in value for k in ["x1", "y1", "x2", "y2"]):
            return float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        a, b, c, d = [float(x) for x in value[:4]]
        if bbox_format == "xywh":
            return a, b, a + c, b + d
        return a, b, c, d
    return None


def _is_invalid_id(gt_id) -> bool:
    try:
        return int(float(gt_id)) < 0
    except Exception:
        return gt_id in [None, "", "None", "null"]


def _append_row(rows: list[dict], camera: str, frame, gt_id, bbox, bbox_format="xyxy"):
    if frame is None or gt_id is None or bbox is None or _is_invalid_id(gt_id):
        return
    parsed = _parse_bbox(bbox, bbox_format=bbox_format)
    if parsed is None:
        return
    x1, y1, x2, y2 = parsed
    if x2 <= x1 or y2 <= y1:
        return
    rows.append({
        "camera": camera,
        "source_frame": int(float(frame)),
        "gt_id": int(float(gt_id)),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "gt_area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
    })


def parse_chirla_annotation_json(annotation_path: str | Path, camera: str, bbox_format="xyxy") -> pd.DataFrame:
    annotation_path = Path(annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows: list[dict] = []

    # Format CHIRLA utama: {"frame_number": [{"id": ..., "BboxP": [x1,y1,x2,y2]}, ...]}
    if isinstance(data, dict):
        for frame_key, objects in data.items():
            try:
                frame_number = int(float(frame_key))
            except Exception:
                continue
            if isinstance(objects, list):
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    bbox = obj.get("BboxP", obj.get("bboxP", obj.get("bbox", obj.get("box"))))
                    gt_id = obj.get("id", obj.get("gt_id", obj.get("person_id", obj.get("track_id"))))
                    _append_row(rows, camera, frame_number, gt_id, bbox, bbox_format="xyxy")

    if not rows:
        def key_to_frame(key):
            try:
                return int(float(key))
            except Exception:
                return None

        def visit(obj: Any, inherited_frame=None):
            if isinstance(obj, dict):
                frame = _get_first(obj, FRAME_KEYS)
                if frame is None:
                    frame = inherited_frame
                bbox = _get_first(obj, BBOX_KEYS)
                gt_id = _get_first(obj, ID_KEYS)
                _append_row(rows, camera, frame, gt_id, bbox, bbox_format=bbox_format)
                for key in LIST_KEYS:
                    if key in obj:
                        visit(obj[key], inherited_frame=frame)
                for k, v in obj.items():
                    if isinstance(v, (list, dict)):
                        next_frame = frame
                        if next_frame is None:
                            next_frame = key_to_frame(k)
                        visit(v, inherited_frame=next_frame)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item, inherited_frame=inherited_frame)

        visit(data)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(["camera", "source_frame", "gt_id", "x1", "y1", "x2", "y2"]).reset_index(drop=True)
    return df


def load_case_ground_truth_debug(case: dict):
    c = _case(case)
    debug = {
        "annotation_dir": str(c.get("annotation_dir")),
        "source_frame_start": c.get("source_frame_start"),
        "source_frame_end": c.get("source_frame_end"),
        "local_frame_start": c.get("local_frame_start", 0),
        "annotation_frame_start": c.get("annotation_frame_start", 1),
        "bbox_format": c.get("bbox_format", "xyxy"),
        "found_files": [],
        "missing_cameras": [],
        "rows_before_filter": 0,
        "rows_after_filter": 0,
        "gt_source_frame_min_before_filter": None,
        "gt_source_frame_max_before_filter": None,
        "gt_source_frame_min_after_filter": None,
        "gt_source_frame_max_after_filter": None,
        "error": None,
    }
    if "annotation_dir" not in c or "source_frame_start" not in c or "source_frame_end" not in c:
        debug["error"] = "Metadata GT belum tersedia untuk case ini."
        return pd.DataFrame(columns=EMPTY_GT_COLUMNS), debug

    frames = []
    try:
        for camera in c.get("cameras", []):
            ann_path = find_annotation_file(c, camera)
            if ann_path is None:
                debug["missing_cameras"].append(camera)
                continue
            debug["found_files"].append(str(ann_path))
            df = parse_chirla_annotation_json(ann_path, camera, bbox_format=c.get("bbox_format", "xyxy"))
            if len(df):
                frames.append(df)
        if not frames:
            debug["error"] = "Annotation file tidak ditemukan atau parser tidak menghasilkan baris GT."
            return pd.DataFrame(columns=EMPTY_GT_COLUMNS), debug

        gt = pd.concat(frames, ignore_index=True)
        debug["rows_before_filter"] = int(len(gt))
        if len(gt):
            debug["gt_source_frame_min_before_filter"] = int(gt["source_frame"].min())
            debug["gt_source_frame_max_before_filter"] = int(gt["source_frame"].max())

        if int(c.get("annotation_frame_start", 1)) == 0:
            gt["source_frame"] = gt["source_frame"].astype(int) + 1

        start = int(c["source_frame_start"])
        end = int(c["source_frame_end"])
        gt = gt[(gt["source_frame"] >= start) & (gt["source_frame"] <= end)].copy()
        debug["rows_after_filter"] = int(len(gt))
        if len(gt):
            debug["gt_source_frame_min_after_filter"] = int(gt["source_frame"].min())
            debug["gt_source_frame_max_after_filter"] = int(gt["source_frame"].max())
        else:
            debug["error"] = "Annotation terbaca, tetapi kosong setelah filter source_frame. Periksa source_frame_start/source_frame_end."
        return gt.reset_index(drop=True), debug
    except Exception as e:
        debug["error"] = str(e)
        return pd.DataFrame(columns=EMPTY_GT_COLUMNS), debug


def load_case_ground_truth(case: dict) -> pd.DataFrame:
    gt, _ = load_case_ground_truth_debug(case)
    return gt


def compute_iou(box_a, box_b) -> float:
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_predictions_to_gt(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold=0.50) -> pd.DataFrame:
    pred = pred_df.copy()
    if len(pred) == 0:
        pred["gt_id"] = []
        pred["gt_iou"] = []
        pred["is_matched"] = []
        return pred
    pred["gt_id"] = -1
    pred["gt_iou"] = 0.0
    pred["is_matched"] = False
    if gt_df is None or len(gt_df) == 0 or "source_frame" not in pred.columns:
        return pred
    gt_groups = {k: g for k, g in gt_df.groupby(["camera", "source_frame"])}
    for idx, row in pred.iterrows():
        key = (row["camera"], int(row["source_frame"]))
        g = gt_groups.get(key)
        if g is None or len(g) == 0:
            continue
        pbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
        best_iou = 0.0
        best_gt = -1
        for _, gt in g.iterrows():
            iou = compute_iou(pbox, [gt["x1"], gt["y1"], gt["x2"], gt["y2"]])
            if iou > best_iou:
                best_iou = iou
                best_gt = int(gt["gt_id"])
        if best_iou >= float(iou_threshold):
            pred.at[idx, "gt_id"] = best_gt
            pred.at[idx, "gt_iou"] = best_iou
            pred.at[idx, "is_matched"] = True
    return pred


def build_track_summary_with_gt(matched_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if matched_df is None or len(matched_df) == 0:
        return pd.DataFrame()
    if "track_key" not in matched_df.columns:
        matched_df = matched_df.copy()
        matched_df["track_key"] = matched_df["camera"].astype(str) + "_T" + matched_df["track_id"].astype(int).astype(str)
    for tkey, group in matched_df.groupby("track_key"):
        matched = group[group["gt_id"] > 0]
        dominant_gt = -1
        dominant_frames = 0
        purity = 0.0
        if len(matched):
            counts = matched["gt_id"].value_counts()
            dominant_gt = int(counts.index[0])
            dominant_frames = int(counts.iloc[0])
            purity = dominant_frames / max(1, len(matched))
        rows.append({
            "track_key": tkey,
            "camera": group["camera"].iloc[0],
            "track_id": int(group["track_id"].iloc[0]),
            "dominant_gt_id": dominant_gt,
            "num_frames": int(group["frame"].nunique()),
            "matched_frames": int(matched["frame"].nunique()) if len(matched) else 0,
            "dominant_frames": dominant_frames,
            "track_gt_purity": float(purity),
            "mean_gt_iou": float(matched["gt_iou"].mean()) if len(matched) else 0.0,
            "avg_conf": float(group["conf"].mean()) if "conf" in group else 0.0,
            "avg_area": float(group["area"].mean()) if "area" in group else 0.0,
            "first_frame": int(group["frame"].min()),
            "last_frame": int(group["frame"].max()),
        })
    return pd.DataFrame(rows)


def _count_id_switches(matched_df: pd.DataFrame) -> int:
    count = 0
    matched = matched_df[matched_df["gt_id"] > 0].copy()
    for (_, _), g in matched.groupby(["camera", "gt_id"]):
        timeline = (
            g.sort_values("source_frame")
            .groupby("source_frame")["track_key"]
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
        )
        if len(timeline) <= 1:
            continue
        count += int((timeline["track_key"] != timeline["track_key"].shift(1)).sum() - 1)
    return max(0, count)


def evaluate_tracking_with_gt(matched_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    raw = len(matched_df)
    unique_tracks = matched_df["track_key"].nunique() if raw and "track_key" in matched_df.columns else 0
    gt_rows = len(gt_df) if gt_df is not None else 0
    matched = matched_df[matched_df["gt_id"] > 0].copy() if raw else pd.DataFrame()
    covered = matched.drop_duplicates(["camera", "source_frame", "gt_id"]) if len(matched) else pd.DataFrame()
    pred_match_rate = len(matched) / max(1, raw)
    gt_coverage_rate = len(covered) / max(1, gt_rows)
    track_summary = build_track_summary_with_gt(matched_df)
    valid_purity = track_summary[track_summary["matched_frames"] > 0]["track_gt_purity"] if len(track_summary) else pd.Series(dtype=float)
    mean_track_purity = float(valid_purity.mean()) if len(valid_purity) else 0.0
    mean_gt_iou = float(matched["gt_iou"].mean()) if len(matched) else 0.0
    idsw = _count_id_switches(matched_df) if raw else 0
    frag = float(matched.groupby(["camera", "gt_id"])["track_key"].nunique().mean()) if len(matched) else 0.0
    fp_rate = (raw - len(matched)) / max(1, raw)
    fragmentation_score = 1.0 / max(1.0, frag) if len(matched) else 0.0
    score = (
        0.30 * gt_coverage_rate
        + 0.25 * pred_match_rate
        + 0.20 * mean_track_purity
        + 0.20 * mean_gt_iou
        + 0.05 * fragmentation_score
        - 0.05 * fp_rate
        - 0.005 * idsw
    )
    score = max(0.0, min(1.0, score))
    return pd.DataFrame([{
        "raw_detections": int(raw),
        "unique_tracks": int(unique_tracks),
        "gt_rows": int(gt_rows),
        "gt_coverage_rate": float(gt_coverage_rate),
        "pred_match_rate": float(pred_match_rate),
        "mean_track_purity": float(mean_track_purity),
        "mean_gt_iou": float(mean_gt_iou),
        "id_switch_count": int(idsw),
        "fragmentation_avg": float(frag),
        "fragmentation_score": float(fragmentation_score),
        "false_positive_rate": float(fp_rate),
        "score": float(score),
    }])


def build_tracking_standard_metrics(
    matched_df: pd.DataFrame | None,
    gt_df: pd.DataFrame | None,
    tracking_eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    gt_df = gt_df if gt_df is not None else pd.DataFrame()
    matched_df = matched_df if matched_df is not None else pd.DataFrame()

    if len(gt_df) == 0:
        return pd.DataFrame([{
            "score_available": False,
            "total_gt_rows": 0,
            "total_prediction_rows": int(len(matched_df)),
            "tp": 0,
            "fp": int(len(matched_df)),
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "raw_mota_simple": None,
            "mota_simple": None,
            "motp_iou": None,
            "id_switch_count": 0,
            "fragmentation_avg": None,
            "mean_track_purity": None,
            "false_positive_rate": None,
            "identity_coverage_rate": None,
            "note": "GT tidak tersedia; metrik akademik tracking tidak aktif.",
        }])

    total_gt_rows = int(len(gt_df))
    total_prediction_rows = int(len(matched_df))
    matched = matched_df[matched_df["gt_id"] > 0].copy() if len(matched_df) and "gt_id" in matched_df else pd.DataFrame()
    covered = (
        matched.drop_duplicates(["camera", "source_frame", "gt_id"])
        if len(matched) and {"camera", "source_frame", "gt_id"}.issubset(matched.columns)
        else pd.DataFrame()
    )

    tp = int(len(matched))
    fp = int(max(0, total_prediction_rows - tp))
    fn = int(max(0, total_gt_rows - len(covered)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    motp_iou = float(matched["gt_iou"].mean()) if len(matched) and "gt_iou" in matched else 0.0

    eval_row = {}
    if tracking_eval_df is not None and len(tracking_eval_df):
        eval_row = tracking_eval_df.iloc[0].to_dict()
    id_switch_count = int(eval_row.get("id_switch_count", 0) or 0)
    fragmentation_avg = float(eval_row.get("fragmentation_avg", 0.0) or 0.0)
    mean_track_purity = float(eval_row.get("mean_track_purity", 0.0) or 0.0)
    false_positive_rate = fp / max(1, total_prediction_rows)

    total_gt_ids = int(gt_df["gt_id"].nunique()) if "gt_id" in gt_df and len(gt_df) else 0
    detected_gt_ids = int(matched["gt_id"].nunique()) if "gt_id" in matched and len(matched) else 0
    identity_coverage_rate = detected_gt_ids / max(1, total_gt_ids)

    raw_mota_simple = 1.0 - ((fn + fp + id_switch_count) / max(1, total_gt_rows))
    mota_simple = max(0.0, min(1.0, raw_mota_simple))

    return pd.DataFrame([{
        "score_available": True,
        "total_gt_rows": total_gt_rows,
        "total_prediction_rows": total_prediction_rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "raw_mota_simple": float(raw_mota_simple),
        "mota_simple": float(mota_simple),
        "motp_iou": float(motp_iou),
        "id_switch_count": id_switch_count,
        "fragmentation_avg": fragmentation_avg,
        "mean_track_purity": mean_track_purity,
        "false_positive_rate": float(false_positive_rate),
        "identity_coverage_rate": float(identity_coverage_rate),
        "note": "Metrik akademik berbasis GT. MOTA sederhana memakai FN, FP, dan ID switch dari evaluasi tracking.",
    }])


def build_gt_coverage_detail(matched_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    if gt_df is None or len(gt_df) == 0:
        return pd.DataFrame()
    gt_summary = gt_df.groupby(["camera", "gt_id"], as_index=False).agg(
        gt_frames=("source_frame", "nunique"),
        first_source_frame=("source_frame", "min"),
        last_source_frame=("source_frame", "max"),
    )
    matched = matched_df[matched_df["gt_id"] > 0].copy()
    if len(matched):
        det = matched.drop_duplicates(["camera", "source_frame", "gt_id", "track_key"]).groupby(["camera", "gt_id"], as_index=False).agg(
            detected_frames=("source_frame", "nunique"),
            tracks=("track_key", "nunique"),
            mean_iou=("gt_iou", "mean"),
            mean_conf=("conf", "mean"),
        )
    else:
        det = pd.DataFrame(columns=["camera", "gt_id", "detected_frames", "tracks", "mean_iou", "mean_conf"])
    out = gt_summary.merge(det, on=["camera", "gt_id"], how="left")
    out["detected_frames"] = out["detected_frames"].fillna(0).astype(int)
    out["tracks"] = out["tracks"].fillna(0).astype(int)
    out["mean_iou"] = out["mean_iou"].fillna(0.0)
    out["mean_conf"] = out["mean_conf"].fillna(0.0)
    out["coverage_per_gt"] = out["detected_frames"] / out["gt_frames"].clip(lower=1)
    return out.sort_values(["camera", "gt_id"]).reset_index(drop=True)


def evaluate_global_id_with_gt(global_track_meta_df: pd.DataFrame, matched_df: pd.DataFrame):
    if global_track_meta_df is None or len(global_track_meta_df) == 0 or matched_df is None or len(matched_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    meta = global_track_meta_df.copy()
    rows = matched_df[matched_df["gt_id"] > 0][["track_key", "camera", "gt_id"]].drop_duplicates()
    joined = meta[["global_id", "track_key", "camera"]].merge(rows, on=["track_key", "camera"], how="left")
    eval_rows = []
    for gid, g in joined.groupby("global_id"):
        gt_ids = sorted({int(x) for x in g["gt_id"].dropna().tolist() if int(x) > 0})
        eval_rows.append({
            "global_id": int(gid),
            "tracks": ", ".join(sorted(g["track_key"].dropna().unique().tolist())),
            "cameras": ", ".join(sorted(g["camera"].dropna().unique().tolist())),
            "gt_ids_inside": str(gt_ids),
            "num_tracks": int(g["track_key"].nunique()),
            "num_cameras": int(g["camera"].nunique()),
            "is_pure_global_id": len(gt_ids) <= 1,
        })
    eval_df = pd.DataFrame(eval_rows).sort_values("global_id") if eval_rows else pd.DataFrame()
    false_merge = int((eval_df["is_pure_global_id"] == False).sum()) if len(eval_df) else 0
    pure_rate = float(eval_df["is_pure_global_id"].mean()) if len(eval_df) else 0.0
    matched_global_ids = int((eval_df["num_tracks"] > 1).sum()) if len(eval_df) else 0
    summary = pd.DataFrame([{
        "total_global_ids": int(len(eval_df)),
        "matched_global_ids": matched_global_ids,
        "pure_global_id_rate": pure_rate,
        "false_merge_count": false_merge,
    }])
    return eval_df, summary


def build_reid_pairwise_evaluation(
    global_meta_df: pd.DataFrame | None,
    track_summary_gt_df: pd.DataFrame | None,
    pair_df: pd.DataFrame | None,
    min_track_purity: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = global_meta_df.copy() if global_meta_df is not None else pd.DataFrame()
    track_gt = track_summary_gt_df.copy() if track_summary_gt_df is not None else pd.DataFrame()
    pairs = pair_df.copy() if pair_df is not None else pd.DataFrame()

    empty_summary = pd.DataFrame([{
        "score_available": False,
        "association_eval_available": False,
        "num_eval_tracks": 0,
        "positive_pair_count": 0,
        "negative_pair_count": 0,
        "predicted_positive_pair_count": 0,
        "tp_pair": 0,
        "fp_pair": 0,
        "fn_pair": 0,
        "tn_pair": 0,
        "pairwise_precision": None,
        "pairwise_recall": None,
        "pairwise_f1": None,
        "false_merge_rate": None,
        "false_split_rate": None,
        "mean_global_id_purity": None,
        "mixed_gid_count": None,
        "expected_global_ids": None,
        "num_global_ids": None,
        "reduction_required": None,
        "reduction_achieved": None,
        "note": "GT track-level data tidak tersedia; pairwise Re-ID evaluation tidak aktif.",
    }])

    if len(meta) == 0 or len(track_gt) == 0:
        return empty_summary, pd.DataFrame()

    needed_gt_cols = {"track_key", "dominant_gt_id", "track_gt_purity"}
    if not needed_gt_cols.issubset(track_gt.columns):
        return empty_summary, pd.DataFrame()

    base_cols = ["track_key", "camera", "global_id", "first_frame", "last_frame"]
    base_cols = [col for col in base_cols if col in meta.columns]
    track_eval = meta[base_cols].merge(
        track_gt[["track_key", "dominant_gt_id", "track_gt_purity"]],
        on="track_key",
        how="left",
    )

    pair_lookup = {}
    if len(pairs):
        for _, row in pairs.iterrows():
            key = tuple(sorted([str(row.get("track_a")), str(row.get("track_b"))]))
            pair_lookup[key] = row.to_dict()

    detail_rows = []
    records = list(track_eval.to_dict("records"))
    for a, b in itertools.combinations(records, 2):
        track_a = str(a["track_key"])
        track_b = str(b["track_key"])
        gt_a = a.get("dominant_gt_id")
        gt_b = b.get("dominant_gt_id")
        purity_a = a.get("track_gt_purity")
        purity_b = b.get("track_gt_purity")

        has_gt = pd.notna(gt_a) and pd.notna(gt_b) and int(gt_a) > 0 and int(gt_b) > 0
        high_purity = (
            has_gt
            and pd.notna(purity_a)
            and pd.notna(purity_b)
            and float(purity_a) >= float(min_track_purity)
            and float(purity_b) >= float(min_track_purity)
        )

        pair_info = pair_lookup.get(tuple(sorted([track_a, track_b])), {})
        same_gt = bool(int(gt_a) == int(gt_b)) if has_gt else None
        same_global = bool(int(a["global_id"]) == int(b["global_id"])) if pd.notna(a.get("global_id")) and pd.notna(b.get("global_id")) else None

        if not has_gt:
            pair_eval_type = "skipped_no_gt"
        elif not high_purity:
            pair_eval_type = "skipped_low_purity"
        elif same_gt and same_global:
            pair_eval_type = "TP_pair"
        elif same_gt and not same_global:
            pair_eval_type = "FN_pair"
        elif (not same_gt) and same_global:
            pair_eval_type = "FP_pair"
        else:
            pair_eval_type = "TN_pair"

        detail_rows.append({
            "track_a": track_a,
            "track_b": track_b,
            "camera_a": a.get("camera"),
            "camera_b": b.get("camera"),
            "global_id_a": a.get("global_id"),
            "global_id_b": b.get("global_id"),
            "dominant_gt_id_a": gt_a,
            "dominant_gt_id_b": gt_b,
            "track_purity_a": purity_a,
            "track_purity_b": purity_b,
            "same_gt": same_gt,
            "same_global_id": same_global,
            "pair_eval_type": pair_eval_type,
            "cosine_similarity": pair_info.get("cosine_similarity"),
            "temporal_gap": pair_info.get("temporal_gap"),
            "temporal_overlap": pair_info.get("temporal_overlap"),
            "merge_status": pair_info.get("merge_status"),
            "merge_reason": pair_info.get("merge_reason"),
            "reject_reason": pair_info.get("merge_reason") if str(pair_info.get("merge_reason", "")).startswith("blocked_") else None,
        })

    detail_df = pd.DataFrame(detail_rows)
    eval_df = detail_df[detail_df["pair_eval_type"].isin(["TP_pair", "FP_pair", "FN_pair", "TN_pair"])].copy() if len(detail_df) else pd.DataFrame()

    num_eval_tracks = int(
        track_eval[
            (track_eval["dominant_gt_id"] > 0)
            & (track_eval["track_gt_purity"] >= float(min_track_purity))
        ]["track_key"].nunique()
    ) if len(track_eval) else 0

    if len(eval_df) == 0:
        summary = empty_summary.copy()
        summary.at[0, "score_available"] = True
        summary.at[0, "num_eval_tracks"] = num_eval_tracks
        summary.at[0, "note"] = "Tidak ada pasangan track valid untuk pairwise Re-ID evaluation."
        return summary, detail_df

    tp_pair = int((eval_df["pair_eval_type"] == "TP_pair").sum())
    fp_pair = int((eval_df["pair_eval_type"] == "FP_pair").sum())
    fn_pair = int((eval_df["pair_eval_type"] == "FN_pair").sum())
    tn_pair = int((eval_df["pair_eval_type"] == "TN_pair").sum())
    positive_pair_count = tp_pair + fn_pair
    negative_pair_count = fp_pair + tn_pair
    predicted_positive_pair_count = tp_pair + fp_pair

    eval_tracks = track_eval[
        (track_eval["dominant_gt_id"] > 0)
        & (track_eval["track_gt_purity"] >= float(min_track_purity))
    ].copy()
    expected_global_ids = int(eval_tracks["dominant_gt_id"].nunique()) if len(eval_tracks) else 0
    num_global_ids = int(eval_tracks["global_id"].nunique()) if len(eval_tracks) else 0
    reduction_required = max(0, int(num_eval_tracks) - expected_global_ids)
    reduction_achieved = max(0, int(num_eval_tracks) - num_global_ids)

    gid_purity_rows = []
    for _, group in eval_tracks.groupby("global_id"):
        gt_ids = {int(x) for x in group["dominant_gt_id"].dropna().tolist() if int(x) > 0}
        gid_purity_rows.append(len(gt_ids) <= 1)
    mixed_gid_count = int(sum(not x for x in gid_purity_rows))
    mean_global_id_purity = float(sum(gid_purity_rows) / max(1, len(gid_purity_rows))) if gid_purity_rows else None

    association_available = positive_pair_count > 0
    if association_available:
        pairwise_precision = tp_pair / max(1, tp_pair + fp_pair)
        pairwise_recall = tp_pair / max(1, tp_pair + fn_pair)
        pairwise_f1 = 2 * pairwise_precision * pairwise_recall / max(1e-9, pairwise_precision + pairwise_recall)
        false_merge_rate = fp_pair / max(1, tp_pair + fp_pair)
        false_split_rate = fn_pair / max(1, tp_pair + fn_pair)
        note = "Pairwise association evaluation tersedia karena ada minimal satu positive GT pair."
    else:
        pairwise_precision = None
        pairwise_recall = None
        pairwise_f1 = None
        false_merge_rate = None
        false_split_rate = None
        note = "No positive association pair available; Re-ID association is not required for this case."

    summary = pd.DataFrame([{
        "score_available": True,
        "association_eval_available": bool(association_available),
        "num_eval_tracks": num_eval_tracks,
        "positive_pair_count": positive_pair_count,
        "negative_pair_count": negative_pair_count,
        "predicted_positive_pair_count": predicted_positive_pair_count,
        "tp_pair": tp_pair,
        "fp_pair": fp_pair,
        "fn_pair": fn_pair,
        "tn_pair": tn_pair,
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f1": pairwise_f1,
        "false_merge_rate": false_merge_rate,
        "false_split_rate": false_split_rate,
        "mean_global_id_purity": mean_global_id_purity,
        "mixed_gid_count": mixed_gid_count,
        "expected_global_ids": expected_global_ids,
        "num_global_ids": num_global_ids,
        "reduction_required": reduction_required,
        "reduction_achieved": reduction_achieved,
        "note": note,
    }])
    return summary, detail_df


def build_gt_debug_info(pred_df: pd.DataFrame, gt_df: pd.DataFrame, matched_df: pd.DataFrame, case: dict, loader_debug: dict | None = None) -> dict:
    c = _case(case)
    info = {
        "annotation_dir": str(c.get("annotation_dir")),
        "source_frame_start": c.get("source_frame_start"),
        "source_frame_end": c.get("source_frame_end"),
        "local_frame_start": c.get("local_frame_start", 0),
        "annotation_frame_start": c.get("annotation_frame_start", 1),
        "bbox_format": c.get("bbox_format", "xyxy"),
        "pred_frame_min": int(pred_df["frame"].min()) if pred_df is not None and len(pred_df) else None,
        "pred_frame_max": int(pred_df["frame"].max()) if pred_df is not None and len(pred_df) else None,
        "pred_source_frame_min": int(pred_df["source_frame"].min()) if pred_df is not None and len(pred_df) and "source_frame" in pred_df else None,
        "pred_source_frame_max": int(pred_df["source_frame"].max()) if pred_df is not None and len(pred_df) and "source_frame" in pred_df else None,
        "gt_source_frame_min": int(gt_df["source_frame"].min()) if gt_df is not None and len(gt_df) else None,
        "gt_source_frame_max": int(gt_df["source_frame"].max()) if gt_df is not None and len(gt_df) else None,
        "pred_cameras": sorted(pred_df["camera"].dropna().unique().tolist()) if pred_df is not None and len(pred_df) else [],
        "gt_cameras": sorted(gt_df["camera"].dropna().unique().tolist()) if gt_df is not None and len(gt_df) else [],
        "pred_rows": int(len(pred_df)) if pred_df is not None else 0,
        "gt_rows": int(len(gt_df)) if gt_df is not None else 0,
        "matched_rows": int((matched_df["gt_id"] > 0).sum()) if matched_df is not None and len(matched_df) else 0,
    }
    if loader_debug is not None:
        info["loader_debug"] = loader_debug
    return info
