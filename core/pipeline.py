from __future__ import annotations

import gc
import json
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.presets import get_camera_stage_config, normalize_config
from config.scoring import SCORE_WEIGHTS
from core.association import assign_global_ids, compute_track_similarity_df
from core.diagnostics import (
    build_reid_diagnostics,
    build_reid_tuning_hints,
    build_tracking_diagnostics,
    build_tracking_tuning_hints,
    save_tuning_hints,
)
from core.evaluation import (
    add_source_frame,
    build_gt_coverage_detail,
    build_reid_pairwise_evaluation,
    build_gt_debug_info,
    build_tracking_standard_metrics,
    build_track_summary_with_gt,
    evaluate_global_id_with_gt,
    evaluate_tracking_with_gt,
    load_case_ground_truth_debug,
    match_predictions_to_gt,
)
from core.gallery import build_global_id_gallery, build_local_track_gallery, build_merge_gallery
from core.models import get_device, load_osnet_model, load_yolo_model
from core.paths import OUTPUT_ROOT
from core.reid import (
    build_sampled_track_crop_df,
    build_track_embedding_df,
    extract_embeddings_from_sampled_df,
)
from core.render import combine_videos_grid, make_streamlit_playable, render_camera_video
from core.tracking import filter_valid_tracks, process_video_tracking
from utils.helpers import ensure_dir, save_json, timestamp_id


def create_run_dir(case: dict) -> Path:
    run_dir = OUTPUT_ROOT / case["case_id"] / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _release_yolo_model(model, use_cuda: bool) -> None:
    """
    Membersihkan state predictor/tracker Ultralytics setelah satu kamera selesai.
    Ini penting untuk multi-camera agar tracker camera sebelumnya tidak terbawa.
    """
    try:
        if getattr(model, "predictor", None) is not None:
            model.predictor = None
    except Exception:
        pass

    try:
        del model
    except Exception:
        pass

    gc.collect()

    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_track_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if len(df) and "track_key" not in df.columns:
        df["track_key"] = (
            df["camera"].astype(str)
            + "_T"
            + df["track_id"].astype(int).astype(str)
        )

    return df


def _merge_full_track_span(
    track_embedding_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> pd.DataFrame:
    if track_embedding_df is None or len(track_embedding_df) == 0:
        return track_embedding_df
    if valid_df is None or len(valid_df) == 0:
        return track_embedding_df

    required_cols = {"track_key", "frame", "conf", "area"}
    if not required_cols.issubset(valid_df.columns):
        return track_embedding_df

    track_span_df = (
        valid_df.groupby("track_key", as_index=False)
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            num_frames=("frame", "nunique"),
            avg_conf=("conf", "mean"),
            avg_area=("area", "mean"),
        )
    )

    drop_cols = [
        "first_frame",
        "last_frame",
        "num_frames",
        "avg_conf",
        "avg_area",
    ]

    out = track_embedding_df.drop(columns=drop_cols, errors="ignore").merge(
        track_span_df,
        on="track_key",
        how="left",
    )

    for col in ["first_frame", "last_frame", "num_frames"]:
        if col in out.columns:
            out[col] = out[col].astype("Int64")

    return out


def _empty_tracking_score(valid_track_count: int, runtime_sec: float = 0.0, reason: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "score_available": False,
        "tracking_score": None,
        "identity_coverage_rate": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "mean_track_purity": None,
        "fragmentation_quality": None,
        "false_positive_quality": None,
        "mixed_track_quality": None,
        "mixed_track_count": None,
        "reason": reason or "GT tidak tersedia; hanya operational summary yang ditampilkan.",
        "num_valid_tracks": int(valid_track_count),
        "runtime_sec": round(float(runtime_sec or 0.0), 2),
    }])


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _weighted_score(values: dict, weights: dict) -> float:
    total_weight = sum(float(v) for v in weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(_clamp01(values.get(key, 0.0)) * float(weight) for key, weight in weights.items()) / total_weight


def _tracking_metrics_for_score(
    tracking_eval_df: pd.DataFrame | None,
    gt_coverage_df: pd.DataFrame | None,
    track_summary_gt_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if tracking_eval_df is None or len(tracking_eval_df) == 0:
        return pd.DataFrame()
    row = tracking_eval_df.iloc[0].to_dict()
    coverage = gt_coverage_df if gt_coverage_df is not None else pd.DataFrame()
    track_gt = track_summary_gt_df if track_summary_gt_df is not None else pd.DataFrame()
    detected_gt_ids = int((coverage.get("detected_frames", pd.Series(dtype=int)) > 0).sum()) if len(coverage) else 0
    total_gt_ids = int(len(coverage)) if len(coverage) else 0
    precision = float(row.get("pred_match_rate", 0.0))
    recall = float(row.get("gt_coverage_rate", 0.0))
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mixed = int((track_gt.get("track_gt_purity", pd.Series(dtype=float)) < 1.0).sum()) if len(track_gt) else 0
    return pd.DataFrame([{
        "detected_gt_ids": detected_gt_ids,
        "identity_coverage_rate": detected_gt_ids / max(1, total_gt_ids),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_track_purity": float(row.get("mean_track_purity", 0.0)),
        "fragmentation_avg": float(row.get("fragmentation_avg", 0.0)),
        "false_positive_rate": float(row.get("false_positive_rate", 0.0)),
        "mixed_track_count": mixed,
    }])


def _tracking_score_df(
    valid_tracks_df: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    runtime_sec: float = 0.0,
    gt_available: bool = False,
    valid_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    valid = valid_tracks_df if valid_tracks_df is not None else pd.DataFrame()
    summary = valid_summary_df if valid_summary_df is not None else pd.DataFrame()
    if len(summary) and {"track_key", "is_valid"}.issubset(summary.columns):
        valid_track_count = int(summary[_truthy_series(summary["is_valid"])]["track_key"].nunique())
    else:
        valid_track_count = int(valid["track_key"].nunique()) if len(valid) and "track_key" in valid else 0
    if metrics_df is None or len(metrics_df) == 0 or not gt_available:
        return _empty_tracking_score(valid_track_count, runtime_sec)

    row = metrics_df.iloc[0].to_dict()
    fragmentation_avg = float(row.get("fragmentation_avg", 0.0))
    mixed_count = int(row.get("mixed_track_count", 0))
    score_inputs = {
        "identity_coverage_rate": row.get("identity_coverage_rate", 0.0),
        "precision": row.get("precision", 0.0),
        "recall": row.get("recall", 0.0),
        "f1": row.get("f1", 0.0),
        "mean_track_purity": row.get("mean_track_purity", 0.0),
        "fragmentation_quality": 1.0 / max(1.0, fragmentation_avg),
        "false_positive_quality": 1.0 - float(row.get("false_positive_rate", 0.0)),
        "mixed_track_quality": 1.0 - (mixed_count / max(1, valid_track_count)),
    }
    score = _weighted_score(score_inputs, SCORE_WEIGHTS["tracking"])
    return pd.DataFrame([{
        "score_available": True,
        "tracking_score": score,
        **{key: _clamp01(value) for key, value in score_inputs.items()},
        "mixed_track_count": mixed_count,
        "num_valid_tracks": valid_track_count,
        "runtime_sec": round(float(runtime_sec or 0.0), 2),
    }])


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "valid"])


def _valid_summary_only(summary_df: pd.DataFrame | None) -> pd.DataFrame:
    if summary_df is None or len(summary_df) == 0:
        return pd.DataFrame()
    if "is_valid" not in summary_df.columns:
        return summary_df.copy()
    return summary_df[_truthy_series(summary_df["is_valid"])].copy()


def _sync_dirty_single_camera_config(config: dict, selected_cameras: list[str]) -> dict:
    if len(selected_cameras) != 1:
        return config
    custom = config.get("custom") or {}
    if not custom.get("tracking_config_dirty"):
        return config
    camera = selected_cameras[0]
    out = copy.deepcopy(config)
    camera_cfg = out.setdefault("camera_configs", {}).setdefault(camera, {})
    camera_cfg["tracking"] = copy.deepcopy(out.get("tracking", {}))
    camera_cfg["filter"] = copy.deepcopy(out.get("filter", {}))
    return out


def _tracking_fingerprint(tracking_cfg: dict) -> str:
    payload = json.dumps(tracking_cfg or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _camera_status_rows(case: dict, run_dir: Path, camera_configs: dict, cameras_done: set[str]) -> pd.DataFrame:
    rows = []
    for camera in case["cameras"]:
        cam_dir = run_dir / camera
        valid_path = cam_dir / "local_tracks_valid.csv"
        raw_path = cam_dir / "local_tracks.csv"
        status = "done" if camera in cameras_done and valid_path.exists() else "not_run"
        score_df = _read_csv_if_exists(cam_dir / "flow1_tracking_score.csv")
        score = score_df.iloc[0].get("tracking_score") if len(score_df) and "tracking_score" in score_df else None
        valid_df = _read_csv_if_exists(valid_path)
        rows.append({
            "camera": camera,
            "status": status,
            "local_tracks_exists": raw_path.exists(),
            "valid_tracks_exists": valid_path.exists(),
            "num_valid_tracks": int(valid_df["track_key"].nunique()) if len(valid_df) and "track_key" in valid_df else 0,
            "tracking_score": score,
            "profile": camera_configs.get(camera, {}).get("profile"),
            "config_mode": camera_configs.get(camera, {}).get("config_mode", "uniform"),
        })
    return pd.DataFrame(rows)


def _evaluate_tracking_outputs(
    case: dict,
    run_dir: Path,
    raw_tracks_all: pd.DataFrame,
    valid_tracks_all: pd.DataFrame,
    valid_summary_all: pd.DataFrame,
    gt_df: pd.DataFrame,
    gt_loader_debug: dict,
) -> dict:
    valid_eval_df = add_source_frame(valid_tracks_all, case)
    valid_eval_df = _build_track_key(valid_eval_df)

    if len(gt_df):
        matched_df = match_predictions_to_gt(
            valid_eval_df,
            gt_df,
            iou_threshold=float(case.get("gt_iou_threshold", 0.50)),
        )
        matched_df = _build_track_key(matched_df)
        tracking_eval_df = evaluate_tracking_with_gt(matched_df, gt_df)
        gt_coverage_df = build_gt_coverage_detail(matched_df, gt_df)
        track_summary_gt_df = build_track_summary_with_gt(matched_df)
        tracking_standard_metrics_df = build_tracking_standard_metrics(
            matched_df,
            gt_df,
            tracking_eval_df,
        )
        gt_debug = build_gt_debug_info(
            valid_eval_df,
            gt_df,
            matched_df,
            case,
            loader_debug=gt_loader_debug,
        )
        matched_df.to_csv(run_dir / "tracking_rows_with_gt.csv", index=False)
        tracking_eval_df.to_csv(run_dir / "tracking_evaluation.csv", index=False)
        gt_coverage_df.to_csv(run_dir / "gt_coverage_detail.csv", index=False)
        track_summary_gt_df.to_csv(run_dir / "track_summary_with_gt.csv", index=False)
        tracking_standard_metrics_df.to_csv(run_dir / "tracking_standard_metrics.csv", index=False)
        eval_outputs = {
            "gt_available": True,
            "gt_df": gt_df,
            "matched_df": matched_df,
            "tracking_eval_df": tracking_eval_df,
            "gt_coverage_df": gt_coverage_df,
            "track_summary_gt_df": track_summary_gt_df,
            "tracking_standard_metrics_df": tracking_standard_metrics_df,
            "gt_debug": gt_debug,
        }
    else:
        tracking_standard_metrics_df = build_tracking_standard_metrics(
            pd.DataFrame(),
            gt_df,
            pd.DataFrame(),
        )
        tracking_standard_metrics_df.to_csv(run_dir / "tracking_standard_metrics.csv", index=False)
        gt_debug = build_gt_debug_info(
            valid_eval_df,
            gt_df,
            pd.DataFrame(),
            case,
            loader_debug=gt_loader_debug,
        )
        eval_outputs = {
            "gt_available": False,
            "tracking_standard_metrics_df": tracking_standard_metrics_df,
            "gt_debug": gt_debug,
        }

    tracking_diagnostics_df = build_tracking_diagnostics(
        raw_tracks_df=raw_tracks_all,
        valid_tracks_df=valid_tracks_all,
        valid_summary_df=valid_summary_all,
        tracking_eval_df=eval_outputs.get("tracking_eval_df"),
        gt_coverage_df=eval_outputs.get("gt_coverage_df"),
    )
    tracking_diagnostics_df.to_csv(run_dir / "tracking_diagnostics.csv", index=False)
    tracking_hints = build_tracking_tuning_hints({"tracking_diagnostics_df": tracking_diagnostics_df})
    pd.DataFrame(tracking_hints).to_csv(run_dir / "tracking_tuning_hints.csv", index=False)
    save_tuning_hints(run_dir / "tuning_hints.json", tracking_hints)

    metrics_for_score = _tracking_metrics_for_score(
        eval_outputs.get("tracking_eval_df"),
        eval_outputs.get("gt_coverage_df"),
        eval_outputs.get("track_summary_gt_df"),
    )
    tracking_score_df = _tracking_score_df(
        valid_tracks_all,
        metrics_for_score,
        runtime_sec=0.0,
        gt_available=eval_outputs.get("gt_available", False),
    )
    tracking_score_df.to_csv(run_dir / "flow1_tracking_score.csv", index=False)

    return {
        "tracking_diagnostics_df": tracking_diagnostics_df,
        "tracking_tuning_hints": tracking_hints,
        "tracking_score_df": tracking_score_df,
        **eval_outputs,
    }


def run_tracking_stage(
    case: dict,
    config: dict,
    yolo_weight=None,
    use_cuda: bool = True,
    run_dir: str | Path | None = None,
    progress_callback=None,
    cameras_to_run: list[str] | None = None,
) -> dict:
    run_dir = Path(run_dir) if run_dir else create_run_dir(case)
    ensure_dir(run_dir)

    config_norm = normalize_config(config)
    selected_cameras = list(cameras_to_run) if cameras_to_run else list(case["cameras"])
    config_norm = _sync_dirty_single_camera_config(config_norm, selected_cameras)
    if len(case.get("cameras", [])) > 3:
        raise ValueError("Multi-camera tracking mendukung maksimal 3 kamera/video.")

    expected_camera_tracking = {}
    for camera in case["cameras"]:
        tracking_cfg, _ = get_camera_stage_config(config_norm, camera)
        cam_dir = run_dir / camera
        expected_camera_tracking[camera] = {
            "tracking": tracking_cfg,
            "tracking_fingerprint": _tracking_fingerprint(tracking_cfg),
            "tracker_yaml_path": str(cam_dir / "generated_tracker_config_used.yaml"),
            "tracking_runtime_manifest_path": str(cam_dir / "tracking_runtime_manifest.yaml"),
        }

    run_config = {
        "case_id": case.get("case_id"),
        "tracking_config_mode": config_norm.get("tracking_config_mode", "uniform"),
        "tracking_preset": config_norm.get("tracking_preset"),
        "filter_preset": config_norm.get("filter_preset"),
        "reid_preset": config_norm.get("reid_preset"),
        "tracking_config": config_norm.get("tracking", {}),
        "filter_config": config_norm.get("filter", {}),
        "reid_config": config_norm.get("reid", {}),
        "camera_configs": config_norm.get("camera_configs", {}),
        "camera_tracking_runtime": expected_camera_tracking,
        "tracking_fingerprints": {
            camera: data["tracking_fingerprint"]
            for camera, data in expected_camera_tracking.items()
        },
        "custom": config_norm.get("custom", {}),
        "case": case,
        "raw_config": config,
    }
    save_json(run_config, run_dir / "run_config.json")
    save_json(
        {
            **run_config,
            "tracking": run_config["tracking_config"],
            "filter": run_config["filter_config"],
            "reid": run_config["reid_config"],
        },
        run_dir / "config_used.json",
    )

    gt_df, gt_loader_debug = load_case_ground_truth_debug(case)
    camera_config_summary = {}
    camera_results = {}
    camera_outputs = {}
    camera_runtime = {}
    cameras_done = set()

    for camera in selected_cameras:
        start_camera = __import__("time").perf_counter()
        video_path = Path(case["video_files"][camera])
        cam_dir = ensure_dir(run_dir / camera)
        tracking_cfg, filter_cfg = get_camera_stage_config(config_norm, camera)
        camera_cfg = (config_norm.get("camera_configs") or {}).get(camera, {}) or {}
        profile_key = camera_cfg.get("profile")
        config_mode = "per_camera"
        tracking_runtime = expected_camera_tracking[camera]
        camera_config = {
            "camera": camera,
            "profile": profile_key,
            "config_mode": config_mode,
            "tracking": tracking_cfg,
            "filter": filter_cfg,
            "tracking_fingerprint": tracking_runtime["tracking_fingerprint"],
            "tracker_yaml_path": tracking_runtime["tracker_yaml_path"],
            "tracking_runtime_manifest_path": tracking_runtime["tracking_runtime_manifest_path"],
        }
        camera_config_summary[camera] = camera_config
        save_json(camera_config, cam_dir / "camera_config_used.json")

        # Penting:
        # YOLO model dibuat baru untuk setiap kamera agar state tracker Ultralytics
        # tidak rusak saat pindah video pada case multi-camera.
        yolo_model = load_yolo_model(yolo_weight)

        try:
            df = process_video_tracking(
                yolo_model,
                video_path,
                camera,
                cam_dir,
                tracking_cfg,
                use_cuda=use_cuda,
                progress_callback=progress_callback,
                tracking_fingerprint=tracking_runtime["tracking_fingerprint"],
            )
        finally:
            _release_yolo_model(yolo_model, use_cuda=use_cuda)

        valid_df, summary_df = filter_valid_tracks(df, filter_cfg)
        df = _build_track_key(df)
        valid_df = _build_track_key(valid_df)
        summary_df = _build_track_key(summary_df)
        valid_summary_df = _valid_summary_only(summary_df)

        valid_df.to_csv(cam_dir / "local_tracks_valid.csv", index=False)
        summary_df.to_csv(cam_dir / "valid_track_summary.csv", index=False)
        local_gallery_df = build_local_track_gallery(
            valid_summary_df,
            cam_dir / "gallery_local",
            extra_meta={"profile": profile_key, "config_mode": config_mode},
        )

        cam_gt_df = gt_df[gt_df["camera"] == camera].copy() if len(gt_df) else pd.DataFrame()
        cam_eval = {}
        if len(cam_gt_df):
            valid_eval_df = add_source_frame(valid_df, {**case, "cameras": [camera]})
            valid_eval_df = _build_track_key(valid_eval_df)
            matched_df = match_predictions_to_gt(
                valid_eval_df,
                cam_gt_df,
                iou_threshold=float(case.get("gt_iou_threshold", 0.50)),
            )
            matched_df = _build_track_key(matched_df)
            tracking_eval_df = evaluate_tracking_with_gt(matched_df, cam_gt_df)
            gt_coverage_df = build_gt_coverage_detail(matched_df, cam_gt_df)
            track_summary_gt_df = build_track_summary_with_gt(matched_df)
            tracking_standard_metrics_df = build_tracking_standard_metrics(
                matched_df,
                cam_gt_df,
                tracking_eval_df,
            )
            matched_df.to_csv(cam_dir / "tracking_rows_with_gt.csv", index=False)
            tracking_eval_df.to_csv(cam_dir / "tracking_evaluation.csv", index=False)
            gt_coverage_df.to_csv(cam_dir / "gt_coverage_detail.csv", index=False)
            track_summary_gt_df.to_csv(cam_dir / "track_summary_with_gt.csv", index=False)
            tracking_standard_metrics_df.to_csv(cam_dir / "tracking_standard_metrics.csv", index=False)
            metrics_for_score = _tracking_metrics_for_score(
                tracking_eval_df,
                gt_coverage_df,
                track_summary_gt_df,
            )
            score_df = _tracking_score_df(
                valid_df,
                metrics_for_score,
                gt_available=True,
                valid_summary_df=summary_df,
            )
            cam_eval = {
                "gt_available": True,
                "matched_df": matched_df,
                "tracking_eval_df": tracking_eval_df,
                "gt_coverage_df": gt_coverage_df,
                "track_summary_gt_df": track_summary_gt_df,
                "tracking_standard_metrics_df": tracking_standard_metrics_df,
                "tracking_score_df": score_df,
            }
        else:
            empty_standard = build_tracking_standard_metrics(pd.DataFrame(), cam_gt_df, pd.DataFrame())
            empty_standard.to_csv(cam_dir / "tracking_standard_metrics.csv", index=False)
            pd.DataFrame().to_csv(cam_dir / "tracking_rows_with_gt.csv", index=False)
            pd.DataFrame().to_csv(cam_dir / "gt_coverage_detail.csv", index=False)
            pd.DataFrame().to_csv(cam_dir / "track_summary_with_gt.csv", index=False)
            score_df = _tracking_score_df(
                valid_df,
                pd.DataFrame(),
                gt_available=False,
                valid_summary_df=summary_df,
            )
            cam_eval = {
                "gt_available": False,
                "tracking_standard_metrics_df": empty_standard,
                "tracking_score_df": score_df,
            }

        runtime = __import__("time").perf_counter() - start_camera
        camera_runtime[camera] = runtime
        score_df["runtime_sec"] = round(float(runtime), 2)
        score_df.to_csv(cam_dir / "flow1_tracking_score.csv", index=False)

        cam_diag = build_tracking_diagnostics(
            raw_tracks_df=df,
            valid_tracks_df=valid_df,
            valid_summary_df=summary_df,
            tracking_eval_df=cam_eval.get("tracking_eval_df"),
            gt_coverage_df=cam_eval.get("gt_coverage_df"),
        )
        cam_diag.to_csv(cam_dir / "tracking_diagnostics.csv", index=False)
        cam_hints = build_tracking_tuning_hints({"tracking_diagnostics_df": cam_diag})
        for hint in cam_hints:
            hint["camera"] = camera
        pd.DataFrame(cam_hints).to_csv(cam_dir / "tracking_tuning_hints.csv", index=False)

        camera_outputs[camera] = {
            "local_tracks": cam_dir / "local_tracks.csv",
            "valid_tracks": cam_dir / "local_tracks_valid.csv",
            "valid_summary": cam_dir / "valid_track_summary.csv",
            "tracking_score": cam_dir / "flow1_tracking_score.csv",
            "local_gallery": cam_dir / "gallery_local",
        }
        camera_results[camera] = {
            "camera": camera,
            "run_dir": cam_dir,
            "raw_tracks": df,
            "valid_tracks": valid_df,
            "valid_summary": summary_df,
            "local_gallery_df": local_gallery_df,
            "tracking_diagnostics_df": cam_diag,
            "tracking_tuning_hints": cam_hints,
            "profile": profile_key,
            "config_mode": config_mode,
            "runtime_sec": runtime,
            **cam_eval,
        }

    for camera in case["cameras"]:
        cam_dir = run_dir / camera
        config_path = cam_dir / "camera_config_used.json"
        loaded_camera_config = None
        if camera not in camera_config_summary and config_path.exists():
            try:
                loaded_camera_config = json.loads(config_path.read_text(encoding="utf-8"))
                camera_config_summary[camera] = loaded_camera_config
            except Exception:
                loaded_camera_config = {}
                camera_config_summary[camera] = {}
        else:
            loaded_camera_config = camera_config_summary.get(camera)

        expected_fingerprint = expected_camera_tracking.get(camera, {}).get("tracking_fingerprint")
        loaded_fingerprint = (loaded_camera_config or {}).get("tracking_fingerprint")
        if not loaded_fingerprint and (loaded_camera_config or {}).get("tracking"):
            loaded_fingerprint = _tracking_fingerprint((loaded_camera_config or {}).get("tracking", {}))
        if (
            camera not in selected_cameras
            and expected_fingerprint
            and loaded_fingerprint
            and loaded_fingerprint != expected_fingerprint
        ):
            camera_config_summary.pop(camera, None)
            continue

        raw_path = cam_dir / "local_tracks.csv"
        valid_path = cam_dir / "local_tracks_valid.csv"
        summary_path = cam_dir / "valid_track_summary.csv"
        if raw_path.exists() or valid_path.exists():
            cameras_done.add(camera)
            if camera not in camera_results:
                camera_results[camera] = {
                    "camera": camera,
                    "run_dir": cam_dir,
                    "raw_tracks": _build_track_key(_read_csv_if_exists(raw_path)),
                    "valid_tracks": _build_track_key(_read_csv_if_exists(valid_path)),
                    "valid_summary": _read_csv_if_exists(summary_path),
                    "local_gallery_df": _read_csv_if_exists(cam_dir / "gallery_local" / "local_gallery_index.csv"),
                    "tracking_score_df": _read_csv_if_exists(cam_dir / "flow1_tracking_score.csv"),
                    "tracking_standard_metrics_df": _read_csv_if_exists(cam_dir / "tracking_standard_metrics.csv"),
                    "gt_coverage_df": _read_csv_if_exists(cam_dir / "gt_coverage_detail.csv"),
                    "track_summary_gt_df": _read_csv_if_exists(cam_dir / "track_summary_with_gt.csv"),
                    "matched_df": _read_csv_if_exists(cam_dir / "tracking_rows_with_gt.csv"),
                    "tracking_diagnostics_df": _read_csv_if_exists(cam_dir / "tracking_diagnostics.csv"),
                    "profile": camera_config_summary.get(camera, {}).get("profile"),
                    "config_mode": camera_config_summary.get(camera, {}).get("config_mode", "uniform"),
                    "runtime_sec": camera_runtime.get(camera, 0.0),
                }

    compact_camera_configs = {
        camera: {
            "profile": cfg.get("profile"),
            "tracking": cfg.get("tracking", {}),
            "filter": cfg.get("filter", {}),
            "tracking_fingerprint": cfg.get("tracking_fingerprint"),
            "tracker_yaml_path": cfg.get("tracker_yaml_path"),
            "tracking_runtime_manifest_path": cfg.get("tracking_runtime_manifest_path"),
        }
        for camera, cfg in camera_config_summary.items()
    }
    save_json(
        {
            "camera_configs": compact_camera_configs,
            "reid": config_norm.get("reid", {}),
        },
        run_dir / "tracking_config_per_camera.json",
    )

    raw_tracks = []
    all_tracks = []
    valid_summaries = []
    for camera in case["cameras"]:
        result = camera_results.get(camera)
        if not result:
            continue
        raw_df = _build_track_key(result.get("raw_tracks", pd.DataFrame()))
        valid_df = _build_track_key(result.get("valid_tracks", pd.DataFrame()))
        summary_df = result.get("valid_summary", pd.DataFrame())
        if len(raw_df):
            raw_tracks.append(raw_df)
        if len(valid_df):
            all_tracks.append(valid_df)
        if summary_df is not None and len(summary_df):
            valid_summaries.append(summary_df)

    valid_tracks_all = (
        pd.concat(all_tracks, ignore_index=True)
        if all_tracks
        else pd.DataFrame()
    )
    raw_tracks_all = (
        pd.concat(raw_tracks, ignore_index=True)
        if raw_tracks
        else pd.DataFrame()
    )
    valid_tracks_all = _build_track_key(valid_tracks_all)
    raw_tracks_all = _build_track_key(raw_tracks_all)
    raw_tracks_all.to_csv(run_dir / "local_tracks.csv", index=False)
    valid_tracks_all.to_csv(run_dir / "valid_tracks_all.csv", index=False)

    valid_summary_all = (
        pd.concat(valid_summaries, ignore_index=True)
        if valid_summaries
        else pd.DataFrame()
    )
    if len(valid_summary_all) and "camera" in valid_summary_all:
        valid_summary_all["profile"] = valid_summary_all["camera"].map(
            lambda cam: camera_config_summary.get(cam, {}).get("profile")
        )
        valid_summary_all["config_mode"] = valid_summary_all["camera"].map(
            lambda cam: camera_config_summary.get(cam, {}).get("config_mode", config_norm.get("tracking_config_mode", "uniform"))
        )

    valid_summary_all.to_csv(run_dir / "valid_track_summary.csv", index=False)
    valid_summary_all.to_csv(run_dir / "valid_track_summary_by_camera.csv", index=False)

    local_gallery_df = build_local_track_gallery(
        _valid_summary_only(valid_summary_all),
        run_dir / "gallery_local",
    )

    eval_outputs = _evaluate_tracking_outputs(
        case,
        run_dir,
        raw_tracks_all,
        valid_tracks_all,
        valid_summary_all,
        gt_df,
        gt_loader_debug,
    )

    camera_score_rows = []
    camera_standard_rows = []
    camera_coverage_rows = []
    camera_hint_rows = []
    gt_available_camera_count = 0
    for camera in case["cameras"]:
        result = camera_results.get(camera, {})
        score_df = result.get("tracking_score_df", pd.DataFrame())
        standard_df = result.get("tracking_standard_metrics_df", pd.DataFrame())
        coverage_df = result.get("gt_coverage_df", pd.DataFrame())
        hints = result.get("tracking_tuning_hints", [])
        cfg_meta = camera_config_summary.get(camera, {})
        if len(score_df):
            row = score_df.iloc[0].to_dict()
        else:
            row = _empty_tracking_score(0).iloc[0].to_dict()
        row.update({
            "camera": camera,
            "profile": cfg_meta.get("profile"),
            "config_mode": cfg_meta.get("config_mode", config_norm.get("tracking_config_mode", "uniform")),
        })
        camera_score_rows.append(row)
        if len(standard_df):
            srow = standard_df.iloc[0].to_dict()
            if bool(srow.get("score_available", False)):
                gt_available_camera_count += 1
            srow["camera"] = camera
            camera_standard_rows.append(srow)
        if len(coverage_df):
            cov = coverage_df.copy()
            cov["camera"] = camera
            camera_coverage_rows.append(cov)
        for hint in hints:
            camera_hint_rows.append(hint)

    tracking_score_by_camera_df = pd.DataFrame(camera_score_rows)
    if len(tracking_score_by_camera_df):
        cols = ["camera"] + [c for c in tracking_score_by_camera_df.columns if c != "camera"]
        tracking_score_by_camera_df = tracking_score_by_camera_df[cols]
    tracking_score_by_camera_df.to_csv(run_dir / "tracking_score_by_camera.csv", index=False)

    standard_by_camera_df = pd.DataFrame(camera_standard_rows)
    standard_by_camera_df.to_csv(run_dir / "tracking_standard_metrics_by_camera.csv", index=False)

    coverage_by_camera_df = pd.concat(camera_coverage_rows, ignore_index=True) if camera_coverage_rows else pd.DataFrame()
    coverage_by_camera_df.to_csv(run_dir / "gt_coverage_by_camera.csv", index=False)

    tracking_tuning_hints_by_camera_df = pd.DataFrame(camera_hint_rows)
    tracking_tuning_hints_by_camera_df.to_csv(run_dir / "tracking_tuning_hints_by_camera.csv", index=False)

    status_df = _camera_status_rows(case, run_dir, camera_config_summary, cameras_done)
    status_df.to_csv(run_dir / "camera_tracking_status.csv", index=False)

    score_path = run_dir / "flow1_tracking_score.csv"
    aggregate_score_df = _read_csv_if_exists(score_path)
    if len(aggregate_score_df):
        scored = tracking_score_by_camera_df[
            tracking_score_by_camera_df.get("score_available", pd.Series(dtype=object)) == True
        ].copy() if len(tracking_score_by_camera_df) else pd.DataFrame()
        aggregate_score_df["camera_score_mean"] = float(scored["tracking_score"].mean()) if len(scored) else None
        aggregate_score_df["camera_score_min"] = float(scored["tracking_score"].min()) if len(scored) else None
        aggregate_score_df["gt_available_camera_count"] = int(gt_available_camera_count)
        aggregate_score_df["total_camera_count"] = int(len(case.get("cameras", [])))
        aggregate_score_df.to_csv(score_path, index=False)
        eval_outputs["tracking_score_df"] = aggregate_score_df

    return {
        "run_dir": run_dir,
        "raw_tracks_all": raw_tracks_all,
        "valid_tracks_all": valid_tracks_all,
        "valid_summary_all": valid_summary_all,
        "local_gallery_df": local_gallery_df,
        "camera_outputs": camera_outputs,
        "camera_results": camera_results,
        "camera_status_df": status_df,
        "tracking_score_by_camera_df": tracking_score_by_camera_df,
        "tracking_standard_metrics_by_camera_df": standard_by_camera_df,
        "gt_coverage_by_camera_df": coverage_by_camera_df,
        "tracking_tuning_hints_by_camera_df": tracking_tuning_hints_by_camera_df,
        "tracking_diagnostics_df": eval_outputs["tracking_diagnostics_df"],
        "tracking_tuning_hints": eval_outputs["tracking_tuning_hints"],
        **eval_outputs,
    }


def run_reid_stage(
    case: dict,
    config: dict,
    run_dir: str | Path,
    osnet_weight=None,
    use_cuda: bool = True,
    reid_progress_callback=None,
    render_progress_callback=None,
    render: bool = True,
    combined_layout: str = "fullscreen_grid",
) -> dict:
    run_dir = Path(run_dir)
    valid_path = run_dir / "valid_tracks_all.csv"

    if not valid_path.exists():
        raise FileNotFoundError(
            "valid_tracks_all.csv tidak ditemukan. Jalankan Tracking terlebih dahulu."
        )

    valid_df = pd.read_csv(valid_path)
    valid_df = _build_track_key(valid_df)
    stage_log = ["Tracking reused"]

    config_norm = normalize_config(config)
    reid_config_used = copy.deepcopy(config_norm.get("reid", {}))
    raw_config_used = copy.deepcopy(config_norm)
    raw_config_used["reid"] = copy.deepcopy(reid_config_used)
    run_config_path = run_dir / "run_config.json"
    run_config = {}
    if run_config_path.exists():
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            run_config = {}
    run_config.update(
        {
            "case_id": case.get("case_id"),
            "tracking_preset": config_norm.get("tracking_preset"),
            "filter_preset": config_norm.get("filter_preset"),
            "reid_preset": config_norm.get("reid_preset"),
            "tracking_config": config_norm.get("tracking", {}),
            "filter_config": config_norm.get("filter", {}),
            "reid_config": reid_config_used,
            "reid_config_used": reid_config_used,
            "custom": config_norm.get("custom", {}),
            "case": case,
            "raw_config": raw_config_used,
        }
    )
    save_json(run_config, run_config_path)
    save_json(reid_config_used, run_dir / "reid_config_used.json")
    config_used_path = run_dir / "config_used.json"
    config_used = {}
    if config_used_path.exists():
        try:
            config_used = json.loads(config_used_path.read_text(encoding="utf-8"))
        except Exception:
            config_used = {}
    config_used.update(
        {
            "reid_config": reid_config_used,
            "reid_config_used": reid_config_used,
            "reid": reid_config_used,
            "custom": config_norm.get("custom", {}),
            "raw_config": raw_config_used,
        }
    )
    save_json(config_used, config_used_path)

    track_embedding_path = run_dir / "track_embedding_df.pkl"
    track_features_path = run_dir / "track_features.npy"
    if track_embedding_path.exists() and track_features_path.exists():
        sampled_df = _read_csv_if_exists(run_dir / "sampled_crop_df.csv")
        track_embedding_df = pd.read_pickle(track_embedding_path)
        track_embedding_df = _merge_full_track_span(track_embedding_df, valid_df)
        track_features = np.load(track_features_path)
        stage_log.append("Embeddings reused")
    else:
        sampled_df = build_sampled_track_crop_df(valid_df, config_norm["filter"])
        sampled_df.to_csv(run_dir / "sampled_crop_df.csv", index=False)

        device = get_device(prefer_cuda=use_cuda)
        osnet_model = load_osnet_model(osnet_weight, device=device)

        crop_rows_df, crop_features = extract_embeddings_from_sampled_df(
            osnet_model,
            sampled_df,
            use_cuda=use_cuda,
            batch_size=32,
            progress_callback=reid_progress_callback,
        )

        crop_rows_df.to_pickle(run_dir / "crop_embedding_df.pkl")
        np.save(run_dir / "crop_features.npy", crop_features)

        track_embedding_df, track_features = build_track_embedding_df(
            crop_rows_df,
            crop_features,
        )
        track_embedding_df = _merge_full_track_span(track_embedding_df, valid_df)

        track_embedding_df.to_pickle(track_embedding_path)
        track_embedding_df.to_csv(run_dir / "track_embedding_summary.csv", index=False)
        np.save(track_features_path, track_features)
        stage_log.append("Embeddings recomputed")

    pair_df = compute_track_similarity_df(track_embedding_df, track_features)
    stage_log.append("Re-ID recomputed")

    global_meta_df, pair_df = assign_global_ids(
        track_embedding_df,
        pair_df,
        reid_config_used,
    )

    pair_df.to_csv(run_dir / "pair_similarity.csv", index=False)
    merged_pairs_df = (
        pair_df[pair_df["merge_status"] == True].copy()
        if len(pair_df) and "merge_status" in pair_df
        else pd.DataFrame()
    )
    merged_pairs_df.to_csv(run_dir / "merged_pairs.csv", index=False)
    global_meta_df.to_csv(run_dir / "global_track_meta.csv", index=False)
    global_meta_df.to_pickle(run_dir / "global_track_df.pkl")

    merge_gallery_df = build_merge_gallery(
        merged_pairs_df,
        global_meta_df,
        run_dir / "merge_gallery",
    )

    global_gallery_df = build_global_id_gallery(
        global_meta_df,
        run_dir / "gallery_global",
    )

    render_df = valid_df.merge(
        global_meta_df[["track_key", "global_id"]],
        on="track_key",
        how="left",
    )

    gt_path = run_dir / "tracking_rows_with_gt.csv"
    matched_df = None

    if gt_path.exists():
        matched_df = pd.read_csv(gt_path)
        matched_df = _build_track_key(matched_df)

        gt_cols = [
            "camera",
            "frame",
            "track_key",
            "gt_id",
            "gt_iou",
            "is_matched",
        ]
        existing_cols = [col for col in gt_cols if col in matched_df.columns]

        render_df = render_df.merge(
            matched_df[existing_cols],
            on=["camera", "frame", "track_key"],
            how="left",
        )

    render_outputs = (
        render_reid_outputs(case, run_dir, render_df, render_progress_callback, combined_layout=combined_layout)
        if render
        else {"rendered_paths": [], "combined_path": None}
    )

    global_eval_df = pd.DataFrame()
    global_summary_df = pd.DataFrame()

    if matched_df is not None and len(matched_df):
        global_eval_df, global_summary_df = evaluate_global_id_with_gt(
            global_meta_df,
            matched_df,
        )
        global_eval_df.to_csv(run_dir / "global_id_evaluation.csv", index=False)
        global_summary_df.to_csv(
            run_dir / "global_id_summary_metrics.csv",
            index=False,
        )

        track_summary_gt_df = build_track_summary_with_gt(matched_df)
        reid_pairwise_eval_df, reid_pairwise_detail_df = build_reid_pairwise_evaluation(
            global_meta_df,
            track_summary_gt_df,
            pair_df,
        )
    else:
        reid_pairwise_eval_df, reid_pairwise_detail_df = build_reid_pairwise_evaluation(
            global_meta_df,
            pd.DataFrame(),
            pair_df,
        )

    reid_pairwise_eval_df.to_csv(run_dir / "reid_pairwise_evaluation.csv", index=False)
    reid_pairwise_detail_df.to_csv(run_dir / "reid_pairwise_detail.csv", index=False)

    reid_diagnostics_df = build_reid_diagnostics(
        track_embedding_df=track_embedding_df,
        pair_df=pair_df,
        global_meta_df=global_meta_df,
        global_summary_df=global_summary_df,
    )
    reid_diagnostics_df.to_csv(run_dir / "reid_diagnostics.csv", index=False)

    reid_result_seed = {
        "reid_diagnostics_df": reid_diagnostics_df,
    }
    reid_hints = build_reid_tuning_hints(reid_result_seed, config_norm)
    pd.DataFrame(reid_hints).to_csv(run_dir / "reid_tuning_hints.csv", index=False)

    existing_hints = []
    hints_path = run_dir / "tuning_hints.json"
    if hints_path.exists():
        try:
            existing_hints = json.loads(hints_path.read_text(encoding="utf-8"))
        except Exception:
            existing_hints = []
    save_tuning_hints(hints_path, existing_hints + reid_hints)
    save_json({"stage_log": stage_log}, run_dir / "reid_stage_log.json")
    (run_dir / "reid_stage_log.txt").write_text("\n".join(stage_log), encoding="utf-8")

    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_dir": run_dir,
        "sampled_df": sampled_df,
        "track_embedding_df": track_embedding_df,
        "pair_df": pair_df,
        "global_meta_df": global_meta_df,
        "merge_gallery_df": merge_gallery_df,
        "global_gallery_df": global_gallery_df,
        "render_df": render_df,
        "rendered_paths": render_outputs["rendered_paths"],
        "combined_path": render_outputs["combined_path"],
        "global_eval_df": global_eval_df,
        "global_summary_df": global_summary_df,
        "reid_pairwise_eval_df": reid_pairwise_eval_df,
        "reid_pairwise_detail_df": reid_pairwise_detail_df,
        "reid_diagnostics_df": reid_diagnostics_df,
        "reid_tuning_hints": reid_hints,
        "reid_config_used": reid_config_used,
        "stage_log": stage_log,
    }


def render_reid_outputs(
    case: dict,
    run_dir: str | Path,
    render_df: pd.DataFrame | None = None,
    render_progress_callback=None,
    combined_layout: str = "fullscreen_grid",
    combined_output_width: int | None = None,
    combined_output_height: int | None = None,
) -> dict:
    run_dir = Path(run_dir)
    if render_df is None:
        valid_path = run_dir / "valid_tracks_all.csv"
        global_path = run_dir / "global_track_meta.csv"
        if not valid_path.exists() or not global_path.exists():
            raise FileNotFoundError(
                "Output tracking/Re-ID belum lengkap. Jalankan Tracking dan Re-ID terlebih dahulu."
            )
        valid_df = _build_track_key(pd.read_csv(valid_path))
        global_meta_df = pd.read_csv(global_path)
        render_df = valid_df.merge(
            global_meta_df[["track_key", "global_id"]],
            on="track_key",
            how="left",
        )

    gt_path = run_dir / "tracking_rows_with_gt.csv"
    if gt_path.exists() and "gt_id" not in render_df.columns:
        matched_df = _build_track_key(pd.read_csv(gt_path))
        gt_cols = ["camera", "frame", "track_key", "gt_id", "gt_iou", "is_matched"]
        existing_cols = [col for col in gt_cols if col in matched_df.columns]
        render_df = render_df.merge(
            matched_df[existing_cols],
            on=["camera", "frame", "track_key"],
            how="left",
        )

    rendered_paths = []
    for camera in case["cameras"]:
        out_path = run_dir / f"{camera}_rendered.mp4"
        rendered = render_camera_video(
            case["video_files"][camera],
            camera,
            render_df,
            out_path,
            label_mode="global",
            show_gt=False,
            progress_callback=render_progress_callback,
        )
        rendered_paths.append(rendered)

    combined_path = None
    if len(rendered_paths) > 1:
        output_width = 1920 if combined_layout == "fullscreen_grid" else 1280 if len(rendered_paths) == 2 else 1440 if len(rendered_paths) == 3 else None
        output_height = 1080 if combined_layout == "fullscreen_grid" else None
        combined_path = combine_videos_grid(
            rendered_paths,
            run_dir / "combined_multicamera_rendered.mp4",
            layout=combined_layout,
            output_width=output_width,
            output_height=output_height,
        )
        combined_path = make_streamlit_playable(combined_path) if combined_path else None
    elif rendered_paths:
        if combined_layout == "fullscreen_grid":
            combined_path = combine_videos_grid(
                rendered_paths,
                run_dir / "combined_multicamera_rendered.mp4",
                layout=combined_layout,
                output_width=1920,
                output_height=1080,
            )
            combined_path = make_streamlit_playable(combined_path) if combined_path else None
        else:
            combined_path = make_streamlit_playable(rendered_paths[0])

    return {"rendered_paths": rendered_paths, "combined_path": combined_path}
