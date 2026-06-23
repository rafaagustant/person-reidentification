from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config.cases import DEMO_CASES
from config.gt_cases import with_gt_meta
from config.presets import (
    VISUAL_PRESETS,
    build_config_from_case_recommendation,
    get_camera_stage_config,
    get_case_camera_configs,
    normalize_config,
)
from config.scoring import SCORE_WEIGHTS
from core.evaluation import load_case_ground_truth_debug
from core.models import cuda_status
from core.paths import OSNET_DEFAULT_PATH, YOLO_LOCAL_PATH
from core.pipeline import render_reid_outputs, run_reid_stage, run_tracking_stage
from core.video_io import get_video_info, read_frame
from ui.components import (
    asset_status,
    download_table,
    hero,
    preview_videos,
    show_gallery,
)
from ui.state import get_case_state, init_state, reset_all, reset_case
from ui.state import (
    clear_all_old_runs,
    clear_current_run_output,
    clear_selected_case_outputs,
    clear_streamlit_cache_and_state,
)
from ui.styles import apply_styles


def _case_label(case: dict) -> str:
    return str(case.get("title", case.get("case_id", "case"))).split("-", 1)[0].strip()


def _safe_key(*parts) -> str:
    text = "_".join(str(part) for part in parts if part is not None)
    for ch in [" ", ".", "/", "\\", ":", "-", "(", ")", "[", "]"]:
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "valid"])


def _init_config_state(case: dict, state: dict) -> None:
    if state.get("config") is not None:
        state["config"] = normalize_config(state["config"])
        return

    state["config"] = _build_effective_case_recommendation(case["case_id"])
    state["config"]["custom"] = {"tracking": False, "filter": False, "reid": False}


def _apply_recommended_config(cfg: dict, case_id: str) -> None:
    recommended = _build_effective_case_recommendation(case_id)
    cfg["tracking"] = copy.deepcopy(recommended["tracking"])
    cfg["filter"] = copy.deepcopy(recommended["filter"])
    cfg["reid"] = copy.deepcopy(recommended["reid"])
    cfg["camera_configs"] = copy.deepcopy(recommended["camera_configs"])
    cfg["recommended_note"] = recommended.get("recommended_note", "")


TRACKING_SOURCE_TO_PRESET = {
    "Preset: Normal / jelas": "normal_clear",
    "Preset: Sedikit orang / jelas": "few_people_clear",
    "Preset: Ramai / banyak orang": "crowded_many_people",
    "Preset: Orang kecil / jauh": "small_distant",
    "Preset: Banyak oklusi / saling menutupi": "occlusion_overlap",
    "Preset: Video blur / gelap": "low_quality_blur",
}


TRACKING_SOURCE_OPTIONS = ["Rekomendasi case", *TRACKING_SOURCE_TO_PRESET.keys()]


LOCAL_CASE_RECOMMENDATION_OVERRIDES = {
    "case_1_normal_success": {
        "camera_configs": {
            "camera_2": {
                "profile": "case_1_final_camera_2",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.45,
                    "min_avg_area": 8000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": False,
            "cross_threshold": 0.75,
            "intra_threshold": 0.80,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 1: single-camera normal, precision tinggi, tanpa fragment recovery.",
        "note": "Config final Case 1: single-camera normal, precision tinggi, tanpa fragment recovery.",
    },
    "case_2_crowded_fragmentation": {
        "camera_configs": {
            "camera_3": {
                "profile": "case_2_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.20,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.20,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.20,
                    "match_thresh": 0.80,
                    "track_buffer": 45,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.40,
                    "min_avg_area": 3000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.78,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 2: crowded/occlusion, filter cukup ketat, intra-camera fragment recovery aktif.",
        "note": "Config final Case 2: crowded/occlusion, filter cukup ketat, intra-camera fragment recovery aktif.",
    },
    "case_3_failure_limitation": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_3_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.03,
                    "yolo_iou": 0.60,
                    "imgsz": 1280,
                    "track_high_thresh": 0.10,
                    "track_low_thresh": 0.03,
                    "new_track_thresh": 0.10,
                    "match_thresh": 0.84,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 15,
                    "min_crops": 15,
                    "min_avg_conf": 0.25,
                    "min_avg_area": 500.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.66,
            "intra_max_gap": 220,
            "intra_max_overlap": 15,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 3: limitation case untuk orang kecil/jauh.",
        "note": "Config final Case 3: limitation case untuk orang kecil/jauh.",
    },
    "case_4_multicamera_success": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_4_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.20,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.20,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.20,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 60,
                    "min_crops": 60,
                    "min_avg_conf": 0.30,
                    "min_avg_area": 2500.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_3": {
                "profile": "case_4_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.15,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.15,
                    "track_low_thresh": 0.04,
                    "new_track_thresh": 0.15,
                    "match_thresh": 0.88,
                    "track_buffer": 90,
                },
                "filter": {
                    "min_frames": 30,
                    "min_crops": 30,
                    "min_avg_conf": 0.27,
                    "min_avg_area": 2000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.74,
            "intra_max_gap": 450,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 4: multi-camera success, 6 local tracks menjadi 3 Global ID.",
        "note": "Config final Case 4: multi-camera success, 6 local tracks menjadi 3 Global ID.",
    },
    "case_5_multicamera_3_video_stress": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_5_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 10,
                    "min_crops": 10,
                    "min_avg_conf": 0.35,
                    "min_avg_area": 3000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_2": {
                "profile": "case_5_final_camera_2",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.45,
                    "min_avg_area": 8000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_3": {
                "profile": "case_5_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.43,
                    "min_avg_area": 8000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.78,
            "intra_max_gap": 450,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 5: stress test 3 kamera, purity tetap 1.0 dan fragment recovery lebih baik.",
        "note": "Config final Case 5: stress test 3 kamera, purity tetap 1.0 dan fragment recovery lebih baik.",
    },
    "case_6_multicamera_temporal_handoff": {
        "camera_configs": {
            "camera_6": {
                "profile": "case_6_final_camera_6",
                "tracking": {
                    "yolo_conf": 0.25,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.07,
                    "new_track_thresh": 0.25,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 45,
                    "min_crops": 45,
                    "min_avg_conf": 0.55,
                    "min_avg_area": 15000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_5": {
                "profile": "case_6_final_camera_5",
                "tracking": {
                    "yolo_conf": 0.25,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.07,
                    "new_track_thresh": 0.25,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 45,
                    "min_crops": 45,
                    "min_avg_conf": 0.55,
                    "min_avg_area": 15000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": False,
            "cross_threshold": 0.80,
            "intra_threshold": 0.00,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "recommended_note": "Config final Case 6: temporal handoff, 6 local tracks menjadi 3 Global ID.",
        "note": "Config final Case 6: temporal handoff, 6 local tracks menjadi 3 Global ID.",
    },
}


def _deep_update_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_effective_case_recommendation(case_id: str) -> dict:
    rec = build_config_from_case_recommendation(case_id)
    override = LOCAL_CASE_RECOMMENDATION_OVERRIDES.get(case_id)
    if override:
        rec = _deep_update_dict(rec, override)
    return rec


def _status_badge(label: str) -> None:
    st.caption(f"Status konfigurasi: **{label}**")


def _tracking_source_status(source: str, dirty: bool) -> str:
    if dirty:
        return "Konfigurasi manual"
    return "Menggunakan rekomendasi case" if source == "Rekomendasi case" else "Menggunakan preset tuning"


def _apply_tracking_source_to_case(cfg: dict, case_id: str, source: str) -> None:
    if source == "Rekomendasi case":
        recommended = _build_effective_case_recommendation(case_id)
        cfg["tracking"] = copy.deepcopy(recommended["tracking"])
        cfg["filter"] = copy.deepcopy(recommended["filter"])
        cfg["camera_configs"] = copy.deepcopy(recommended["camera_configs"])
        cfg["tracking_config_source"] = source
        return
    preset_key = TRACKING_SOURCE_TO_PRESET[source]
    _apply_visual_preset_to_case(cfg, preset_key)
    cfg["tracking_config_source"] = source


def _apply_tracking_source_to_camera(cfg: dict, case_id: str, camera: str, source: str) -> None:
    cfg.setdefault("camera_configs", {}).setdefault(camera, {})
    if source == "Rekomendasi case":
        recommended = _build_effective_case_recommendation(case_id)
        camera_cfg = (recommended.get("camera_configs") or {}).get(camera)
        if camera_cfg:
            cfg["camera_configs"][camera] = copy.deepcopy(camera_cfg)
        else:
            cfg["camera_configs"][camera]["tracking"] = copy.deepcopy(recommended["tracking"])
            cfg["camera_configs"][camera]["filter"] = copy.deepcopy(recommended["filter"])
            cfg["camera_configs"][camera]["profile"] = f"recommended_{camera}"
        cfg["camera_configs"][camera]["tracking_config_source"] = source
        return
    preset_key = TRACKING_SOURCE_TO_PRESET[source]
    _apply_visual_preset_to_camera(cfg, camera, preset_key)
    cfg["camera_configs"][camera]["tracking_config_source"] = source


def _mark_dirty_if_changed(custom: dict, key: str, before: dict, after: dict) -> None:
    if before != after:
        custom[key] = True


def _invalidate_tracking_state(state: dict) -> None:
    """Hapus hasil tahap 1-3 ketika konfigurasi tracking/filter berubah."""
    for key in [
        "tracking_result",
        "camera_status_df",
        "tracking_score_df",
        "reid_result",
        "global_id_result",
        "merge_pair_result",
        "association_eval_result",
        "reid_score_df",
        "final_score_df",
    ]:
        state[key] = None
    state["tracking_done"] = False
    state["reid_done"] = False
    state["render_done"] = False
    state["run_dir"] = None


def _invalidate_reid_state(state: dict) -> None:
    """Hapus hasil tahap 2-3 ketika konfigurasi Re-ID berubah."""
    for key in [
        "reid_result",
        "global_id_result",
        "merge_pair_result",
        "association_eval_result",
        "reid_score_df",
        "final_score_df",
    ]:
        state[key] = None
    state["reid_done"] = False
    state["render_done"] = False


def _normalize_reid_config_values(reid_cfg: dict | None) -> dict:
    reid_cfg = reid_cfg or {}
    return {
        "enable_cross_camera": bool(reid_cfg.get("enable_cross_camera", False)),
        "enable_strict_intra": bool(reid_cfg.get("enable_strict_intra", False)),
        "cross_threshold": float(reid_cfg.get("cross_threshold", 0.75)),
        "intra_threshold": float(reid_cfg.get("intra_threshold", 0.80)),
        "intra_max_gap": int(reid_cfg.get("intra_max_gap", 30)),
        "intra_max_overlap": int(reid_cfg.get("intra_max_overlap", 0)),
        "use_mnn": bool(reid_cfg.get("use_mnn", True)),
    }


def _sync_reid_config_aliases(cfg: dict, reid_cfg: dict, source: str = "Konfigurasi manual") -> dict:
    active_reid = _normalize_reid_config_values(reid_cfg)
    cfg["reid"] = copy.deepcopy(active_reid)
    cfg["reid_config"] = copy.deepcopy(active_reid)
    cfg["reid_config_used"] = copy.deepcopy(active_reid)

    raw_config = cfg.setdefault("raw_config", {})
    raw_config["reid"] = copy.deepcopy(active_reid)
    raw_config["reid_config"] = copy.deepcopy(active_reid)

    custom = cfg.setdefault("custom", {"tracking": False, "filter": False, "reid": False})
    custom["reid_config_source"] = source
    custom["reid"] = source == "Konfigurasi manual"
    custom["reid_config_dirty"] = source == "Konfigurasi manual"
    cfg["custom"] = custom
    return cfg


def _set_runtime_config_from_state(state: dict) -> None:
    cfg = normalize_config(state["config"])
    cfg = _sync_reid_config_aliases(
        cfg,
        state["config"].get("reid", {}),
        state["config"].get("custom", {}).get("reid_config_source", "Rekomendasi case"),
    )
    state["runtime_config"] = cfg


def _sync_single_camera_config(case: dict, cfg: dict) -> None:
    cameras = case.get("cameras", [])
    if len(cameras) != 1:
        return
    camera = cameras[0]
    camera_cfg = cfg.setdefault("camera_configs", {}).setdefault(camera, {})
    camera_cfg["tracking"] = copy.deepcopy(cfg.get("tracking", {}))
    camera_cfg["filter"] = copy.deepcopy(cfg.get("filter", {}))
    camera_cfg.setdefault("profile", f"single_{camera}")


def _widget_set(key: str, value, force: bool = False) -> None:
    if force or key not in st.session_state:
        st.session_state[key] = value


def _tracking_widget_keys(scope: str) -> dict:
    return {
        "yolo_conf": _safe_key(scope, "tracking", "yolo_conf"),
        "yolo_iou": _safe_key(scope, "tracking", "yolo_iou"),
        "imgsz": _safe_key(scope, "tracking", "imgsz"),
        "track_buffer": _safe_key(scope, "tracking", "track_buffer"),
        "track_high_thresh": _safe_key(scope, "tracking", "high"),
        "track_low_thresh": _safe_key(scope, "tracking", "low"),
        "new_track_thresh": _safe_key(scope, "tracking", "new"),
        "match_thresh": _safe_key(scope, "tracking", "match"),
    }


def _filter_widget_keys(scope: str) -> dict:
    return {
        "min_frames": _safe_key(scope, "filter", "min_frames"),
        "min_crops": _safe_key(scope, "filter", "min_crops"),
        "max_samples_per_track": _safe_key(scope, "filter", "max_samples"),
        "min_avg_conf": _safe_key(scope, "filter", "conf"),
        "min_avg_area": _safe_key(scope, "filter", "area"),
        "crop_selection_strategy": _safe_key(scope, "filter", "strategy"),
    }


def _reid_widget_keys(scope: str) -> dict:
    return {
        "enable_cross_camera": _safe_key(scope, "reid", "cross"),
        "enable_strict_intra": _safe_key(scope, "reid", "intra"),
        "use_mnn": _safe_key(scope, "reid", "mnn"),
        "cross_threshold": _safe_key(scope, "reid", "cross_threshold"),
        "intra_threshold": _safe_key(scope, "reid", "intra_threshold"),
        "intra_max_gap": _safe_key(scope, "reid", "gap"),
        "intra_max_overlap": _safe_key(scope, "reid", "overlap"),
    }


def _prime_tracking_widgets(scope: str, cfg: dict, force: bool = False) -> None:
    keys = _tracking_widget_keys(scope)
    imgsz_options = [320, 640, 960, 1280]
    imgsz = int(cfg.get("imgsz", 640))
    if imgsz not in imgsz_options:
        imgsz = 640
    _widget_set(keys["yolo_conf"], float(cfg.get("yolo_conf", 0.25)), force)
    _widget_set(keys["yolo_iou"], float(cfg.get("yolo_iou", 0.50)), force)
    _widget_set(keys["imgsz"], imgsz, force)
    _widget_set(keys["track_buffer"], int(cfg.get("track_buffer", 30)), force)
    _widget_set(keys["track_high_thresh"], float(cfg.get("track_high_thresh", 0.25)), force)
    _widget_set(keys["track_low_thresh"], float(cfg.get("track_low_thresh", 0.05)), force)
    _widget_set(keys["new_track_thresh"], float(cfg.get("new_track_thresh", 0.25)), force)
    _widget_set(keys["match_thresh"], float(cfg.get("match_thresh", 0.80)), force)


def _prime_filter_widgets(scope: str, cfg: dict, force: bool = False) -> None:
    keys = _filter_widget_keys(scope)
    strategy = str(cfg.get("crop_selection_strategy", "quality"))
    if strategy not in ["quality", "uniform"]:
        strategy = "quality"
    _widget_set(keys["min_frames"], int(cfg.get("min_frames", 50)), force)
    _widget_set(keys["min_crops"], int(cfg.get("min_crops", cfg.get("min_frames", 50))), force)
    _widget_set(keys["max_samples_per_track"], int(cfg.get("max_samples_per_track", 32)), force)
    _widget_set(keys["min_avg_conf"], float(cfg.get("min_avg_conf", 0.30)), force)
    _widget_set(keys["min_avg_area"], float(cfg.get("min_avg_area", 2000.0)), force)
    _widget_set(keys["crop_selection_strategy"], strategy, force)


def _prime_reid_widgets(scope: str, cfg: dict, force: bool = False) -> None:
    cfg = _normalize_reid_config_values(cfg)
    keys = _reid_widget_keys(scope)
    _widget_set(keys["enable_cross_camera"], bool(cfg["enable_cross_camera"]), force)
    _widget_set(keys["enable_strict_intra"], bool(cfg["enable_strict_intra"]), force)
    _widget_set(keys["use_mnn"], bool(cfg["use_mnn"]), force)
    _widget_set(keys["cross_threshold"], float(cfg["cross_threshold"]), force)
    _widget_set(keys["intra_threshold"], float(cfg["intra_threshold"]), force)
    _widget_set(keys["intra_max_gap"], int(cfg["intra_max_gap"]), force)
    _widget_set(keys["intra_max_overlap"], int(cfg["intra_max_overlap"]), force)


def _canonical_tracking(cfg: dict) -> dict:
    return {
        "yolo_conf": round(float(cfg.get("yolo_conf", 0.0)), 4),
        "yolo_iou": round(float(cfg.get("yolo_iou", 0.0)), 4),
        "imgsz": int(cfg.get("imgsz", 0)),
        "track_high_thresh": round(float(cfg.get("track_high_thresh", 0.0)), 4),
        "track_low_thresh": round(float(cfg.get("track_low_thresh", 0.0)), 4),
        "new_track_thresh": round(float(cfg.get("new_track_thresh", 0.0)), 4),
        "match_thresh": round(float(cfg.get("match_thresh", 0.0)), 4),
        "track_buffer": int(cfg.get("track_buffer", 0)),
    }


def _canonical_filter(cfg: dict) -> dict:
    return {
        "min_frames": int(cfg.get("min_frames", 0)),
        "min_crops": int(cfg.get("min_crops", 0)),
        "min_avg_conf": round(float(cfg.get("min_avg_conf", 0.0)), 4),
        "min_avg_area": round(float(cfg.get("min_avg_area", 0.0)), 2),
        "max_samples_per_track": int(cfg.get("max_samples_per_track", 0)),
        "crop_selection_strategy": str(cfg.get("crop_selection_strategy", "quality")),
    }


def _recommended_tracking_filter_for_case(case_id: str, camera: str | None = None) -> tuple[dict, dict]:
    rec = _build_effective_case_recommendation(case_id)
    if camera:
        cam_cfg = (rec.get("camera_configs") or {}).get(camera, {})
        return copy.deepcopy(cam_cfg.get("tracking", rec["tracking"])), copy.deepcopy(cam_cfg.get("filter", rec["filter"]))
    return copy.deepcopy(rec["tracking"]), copy.deepcopy(rec["filter"])


def _tracking_filter_base_for_source(case_id: str, source: str, camera: str | None = None) -> tuple[dict, dict]:
    if source == "Rekomendasi case":
        return _recommended_tracking_filter_for_case(case_id, camera)
    preset_key = TRACKING_SOURCE_TO_PRESET.get(source)
    if preset_key and preset_key in VISUAL_PRESETS:
        preset = VISUAL_PRESETS[preset_key]
        return copy.deepcopy(preset["tracking"]), copy.deepcopy(preset["filter"])
    return _recommended_tracking_filter_for_case(case_id, camera)


def _tracking_filter_equal_to_source(
    case_id: str,
    source: str,
    tracking_cfg: dict,
    filter_cfg: dict,
    camera: str | None = None,
) -> bool:
    base_tracking, base_filter = _tracking_filter_base_for_source(case_id, source, camera)
    return _canonical_tracking(tracking_cfg) == _canonical_tracking(base_tracking) and _canonical_filter(filter_cfg) == _canonical_filter(base_filter)


def _source_status(source: str, is_clean: bool) -> str:
    if not is_clean:
        return "Konfigurasi manual"
    return "Menggunakan rekomendasi case" if source == "Rekomendasi case" else "Menggunakan preset tuning"


def _tracking_custom_inputs(scope: str, cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    _prime_tracking_widgets(scope, out, force=False)
    keys = _tracking_widget_keys(scope)

    cols = st.columns(4)
    out["yolo_conf"] = cols[0].number_input("yolo_conf", 0.0, 1.0, step=0.01, key=keys["yolo_conf"])
    out["yolo_iou"] = cols[1].number_input("yolo_iou", 0.0, 1.0, step=0.01, key=keys["yolo_iou"])
    out["imgsz"] = cols[2].selectbox("imgsz", [320, 640, 960, 1280], key=keys["imgsz"])
    out["track_buffer"] = cols[3].number_input("track_buffer", 1, 300, step=1, key=keys["track_buffer"])

    cols = st.columns(4)
    out["track_high_thresh"] = cols[0].number_input("track_high_thresh", 0.0, 1.0, step=0.01, key=keys["track_high_thresh"])
    out["track_low_thresh"] = cols[1].number_input("track_low_thresh", 0.0, 1.0, step=0.01, key=keys["track_low_thresh"])
    out["new_track_thresh"] = cols[2].number_input("new_track_thresh", 0.0, 1.0, step=0.01, key=keys["new_track_thresh"])
    out["match_thresh"] = cols[3].number_input("match_thresh", 0.0, 1.0, step=0.01, key=keys["match_thresh"])

    out["imgsz"] = int(out["imgsz"])
    out["track_buffer"] = int(out["track_buffer"])
    return out


def _filter_custom_inputs(scope: str, cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    _prime_filter_widgets(scope, out, force=False)
    keys = _filter_widget_keys(scope)

    cols = st.columns(3)
    out["min_frames"] = cols[0].number_input("min_frames", 1, 1000, step=1, key=keys["min_frames"])
    out["min_crops"] = cols[1].number_input(
        "min_crops",
        0,
        1000,
        step=1,
        key=keys["min_crops"],
        help="Gunakan 0 hanya untuk diagnosis track yang tidak memiliki crop.",
    )
    out["max_samples_per_track"] = cols[2].number_input("max_samples_per_track", 1, 256, step=1, key=keys["max_samples_per_track"])

    cols = st.columns(3)
    out["min_avg_conf"] = cols[0].number_input("min_avg_conf", 0.0, 1.0, step=0.01, key=keys["min_avg_conf"])
    out["min_avg_area"] = cols[1].number_input("min_avg_area", 0.0, 100000.0, step=100.0, key=keys["min_avg_area"])
    out["crop_selection_strategy"] = cols[2].selectbox("crop_selection_strategy", ["quality", "uniform"], key=keys["crop_selection_strategy"])

    out["min_frames"] = int(out["min_frames"])
    out["min_crops"] = int(out["min_crops"])
    out["max_samples_per_track"] = int(out["max_samples_per_track"])
    return out


def _reid_custom_inputs(scope: str, cfg: dict) -> dict:
    out = _normalize_reid_config_values(copy.deepcopy(cfg))
    _prime_reid_widgets(scope, out, force=False)
    keys = _reid_widget_keys(scope)

    cols = st.columns(3)
    out["enable_cross_camera"] = cols[0].checkbox("enable_cross_camera", key=keys["enable_cross_camera"])
    out["enable_strict_intra"] = cols[1].checkbox("enable_strict_intra", key=keys["enable_strict_intra"])
    out["use_mnn"] = cols[2].checkbox("use_mnn", key=keys["use_mnn"])

    cols = st.columns(4)
    out["cross_threshold"] = cols[0].number_input("cross_threshold", 0.0, 1.0, step=0.01, key=keys["cross_threshold"])
    out["intra_threshold"] = cols[1].number_input("intra_threshold", 0.0, 1.0, step=0.001, format="%.3f", key=keys["intra_threshold"])
    out["intra_max_gap"] = cols[2].number_input("intra_max_gap", 0, 10000, step=1, key=keys["intra_max_gap"])
    out["intra_max_overlap"] = cols[3].number_input("intra_max_overlap", 0, 10000, step=1, key=keys["intra_max_overlap"])
    return _normalize_reid_config_values(out)


def _save_output_table(df: pd.DataFrame | None, run_dir: str | Path | None, filename: str) -> None:
    if df is None or len(df) == 0 or run_dir is None:
        return
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(run_dir) / filename, index=False)


def _show_table(
    df: pd.DataFrame | None,
    title: str,
    filename: str,
    key: str,
    columns: list[str] | None = None,
    run_dir: str | Path | None = None,
) -> pd.DataFrame:
    st.markdown(f"#### {title}")
    if df is None or len(df) == 0:
        st.info("Belum ada data.")
        return pd.DataFrame()
    out = df.copy()
    if columns is not None:
        existing = [col for col in columns if col in out.columns]
        out = out[existing]
    preview = out.head(500)
    if len(out) > len(preview):
        st.caption(f"Menampilkan preview 500 dari {len(out)} baris. Download berisi data penuh.")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    _save_output_table(out, run_dir, filename)
    download_table(out, f"Unduh {filename}", filename, key=key)
    with st.expander("Salin sebagai CSV/TSV", expanded=False):
        fmt = st.radio(
            "Format salin",
            ["CSV", "TSV"],
            horizontal=True,
            key=f"{key}_copy_format",
        )
        sep = "\t" if fmt == "TSV" else ","
        st.code(out.to_csv(index=False, sep=sep), language="text")
    return out


def _download_file(path: str | Path, label: str, key: str) -> None:
    path = Path(path)
    if not path.exists():
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/octet-stream",
        key=key,
    )


def _case_overview(case: dict) -> None:
    st.markdown("### Ringkasan case")
    gt_df, gt_debug = load_case_ground_truth_debug(case)
    gt_available = len(gt_df) > 0
    video_infos = {cam: get_video_info(case["video_files"][cam]) for cam in case["cameras"]}
    durations = [info.get("duration_sec", 0) for info in video_infos.values() if info.get("duration_sec")]
    frame_ranges = [
        f"{int(info.get('frames', 0))} frame"
        for info in video_infos.values()
        if info.get("frames") is not None
    ]

    rows = [{
        "ID case": case.get("case_id"),
        "Skenario": case.get("title"),
        "Kamera": ", ".join(case.get("cameras", [])),
        "Rentang frame": f"{case.get('source_frame_start', '-')}-{case.get('source_frame_end', '-')}",
        "Durasi": round(max(durations), 2) if durations else "-",
        "Jumlah identitas GT": int(gt_df["gt_id"].nunique()) if gt_available else "-",
        "Tantangan": case.get("scene_type", "-"),
    }]
    _show_table(
        pd.DataFrame(rows),
        "Ringkasan case",
        f"{case['case_id']}_case_annotation_summary.csv",
        _safe_key(case["case_id"], "download", "case_summary"),
    )
    with st.expander("Detail metadata case", expanded=False):
        st.json({
            **case,
            "gt_ids": list(map(int, sorted(gt_df["gt_id"].unique()))) if gt_available else [],
            "total_gt_rows": int(len(gt_df)) if gt_available else 0,
            "gt_loader": gt_debug,
        })

    if frame_ranges:
        st.caption("Video frames: " + "; ".join(f"{cam}: {frame_ranges[i]}" for i, cam in enumerate(video_infos)))
    if not gt_available:
        st.info(f"Annotation tidak tersedia atau kosong. Evaluasi berbasis GT tidak aktif. {gt_debug.get('error') or ''}")


def tracking_config_panel(case: dict, state: dict) -> dict:
    case_id = case["case_id"]
    _init_config_state(case, state)
    cfg = normalize_config(state["config"])
    custom = cfg.setdefault("custom", {"tracking": False, "filter": False, "reid": False})

    st.caption("Pilih rekomendasi case atau preset tuning sebagai nilai awal. Jika angka diubah, status otomatis menjadi konfigurasi manual dan backend langsung memakai nilai baru.")
    source = custom.get("tracking_config_source") or cfg.get("tracking_config_source") or "Rekomendasi case"
    selected_source = st.selectbox(
        "Sumber konfigurasi tracking",
        TRACKING_SOURCE_OPTIONS,
        index=TRACKING_SOURCE_OPTIONS.index(source) if source in TRACKING_SOURCE_OPTIONS else 0,
        key=_safe_key(case_id, "tracking_config_source_select"),
    )

    if selected_source != source:
        _apply_tracking_source_to_case(cfg, case_id, selected_source)
        cfg["custom"]["tracking_config_source"] = selected_source
        cfg["custom"]["tracking_config_dirty"] = False
        cfg["custom"]["filter_config_dirty"] = False
        _prime_tracking_widgets(case_id, cfg["tracking"], force=True)
        _prime_filter_widgets(case_id, cfg["filter"], force=True)
        _sync_single_camera_config(case, cfg)
        state["config"] = normalize_config(cfg)
        state["runtime_config"] = normalize_config(cfg)
        _invalidate_tracking_state(state)
        st.rerun()

    cols = st.columns([1, 3])
    if cols[0].button("Reset case ini", key=_safe_key(case_id, "reset_case")):
        reset_case(case_id, delete_output=True)
        st.rerun()

    before_tracking = copy.deepcopy(cfg["tracking"])
    before_filter = copy.deepcopy(cfg["filter"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Parameter tracking")
        cfg["tracking"] = _tracking_custom_inputs(case_id, cfg["tracking"])
    with right:
        st.markdown("#### Parameter filter")
        cfg["filter"] = _filter_custom_inputs(case_id, cfg["filter"])

    is_clean = _tracking_filter_equal_to_source(case_id, selected_source, cfg["tracking"], cfg["filter"])
    custom["tracking_config_source"] = selected_source
    custom["tracking_config_dirty"] = not is_clean
    custom["filter_config_dirty"] = not is_clean

    if before_tracking != cfg["tracking"] or before_filter != cfg["filter"]:
        _invalidate_tracking_state(state)

    cfg["custom"] = custom
    _sync_single_camera_config(case, cfg)
    state["config"] = normalize_config(cfg)
    state["runtime_config"] = normalize_config(state["config"])
    _status_badge(_source_status(selected_source, is_clean))

    if selected_source != "Rekomendasi case":
        preset_key = TRACKING_SOURCE_TO_PRESET[selected_source]
        st.caption(VISUAL_PRESETS[preset_key]["description"])
    elif cfg.get("recommended_note"):
        st.caption(cfg["recommended_note"])

    return state["config"]


def reid_config_panel(case: dict, state: dict) -> dict:
    case_id = case["case_id"]
    _init_config_state(case, state)

    cfg = normalize_config(state["config"])
    custom = cfg.setdefault("custom", {"tracking": False, "filter": False, "reid": False})
    recommended = _build_effective_case_recommendation(case_id)
    recommended_reid = _normalize_reid_config_values(recommended.get("reid", {}))
    current_reid = _normalize_reid_config_values(cfg.get("reid") or recommended_reid)

    st.caption("Ubah nilai Re-ID di form, lalu klik Terapkan config Re-ID. Tracking, crop, dan embedding tidak diulang; hanya hasil Re-ID dan render lama yang dihapus.")
    cols = st.columns([3, 1])
    with cols[0]:
        st.selectbox(
            "Sumber konfigurasi Re-ID",
            ["Rekomendasi case"],
            index=0,
            key=_safe_key(case_id, "reid_config_source_view"),
            disabled=True,
        )
    with cols[1]:
        if st.button("Reset Re-ID", key=_safe_key(case_id, "reset_reid_to_recommendation")):
            cfg = _sync_reid_config_aliases(cfg, recommended_reid, "Rekomendasi case")
            _prime_reid_widgets(case_id, recommended_reid, force=True)
            state["config"] = normalize_config(cfg)
            state["config"] = _sync_reid_config_aliases(state["config"], recommended_reid, "Rekomendasi case")
            _set_runtime_config_from_state(state)
            _invalidate_reid_state(state)
            st.rerun()

    with st.form(key=_safe_key(case_id, "reid_config_form")):
        pending_reid = _reid_custom_inputs(case_id, current_reid)
        pending_reid = _normalize_reid_config_values(pending_reid)
        effective_source = "Rekomendasi case" if pending_reid == recommended_reid else "Konfigurasi manual"
        if effective_source == "Konfigurasi manual":
            st.warning("Nilai Re-ID berbeda dari rekomendasi case. Saat diterapkan, backend akan memakai konfigurasi manual ini.")
        submitted = st.form_submit_button("Terapkan config Re-ID")

    if submitted:
        cfg = _sync_reid_config_aliases(cfg, pending_reid, effective_source)
        state["config"] = normalize_config(cfg)
        state["config"] = _sync_reid_config_aliases(state["config"], pending_reid, effective_source)
        _set_runtime_config_from_state(state)
        _invalidate_reid_state(state)
        st.success("Config Re-ID aktif diperbarui.")
    else:
        state["config"] = normalize_config(cfg)
        _set_runtime_config_from_state(state)

    _status_badge("Menggunakan rekomendasi case" if effective_source == "Rekomendasi case" else "Konfigurasi manual")
    with st.expander("Active Re-ID config saat ini", expanded=False):
        st.json(state["config"]["reid"])

    return state["config"]


def _tracking_summary_df(result: dict, runtime: float | None) -> pd.DataFrame:
    summary = result.get("valid_summary_all", pd.DataFrame()).copy()
    valid = result.get("valid_tracks_all", pd.DataFrame())
    if len(summary) and len(valid) and "track_key" in valid:
        crops = valid.groupby("track_key", as_index=False).agg(crop_count=("crop_path", "count"))
        summary = summary.merge(crops, on="track_key", how="left")
    if result.get("track_summary_gt_df") is not None and len(result["track_summary_gt_df"]):
        gt_cols = ["track_key", "dominant_gt_id", "track_gt_purity"]
        existing = [col for col in gt_cols if col in result["track_summary_gt_df"].columns]
        summary = summary.merge(result["track_summary_gt_df"][existing], on="track_key", how="left")
        summary = summary.rename(columns={"track_gt_purity": "track_purity"})
    return summary


def _valid_track_summary_df(result: dict) -> pd.DataFrame:
    summary = result.get("valid_summary_all", pd.DataFrame()).copy()
    if len(summary) and "is_valid" in summary:
        summary = summary[_truthy_series(summary["is_valid"])].copy()
    valid = result.get("valid_tracks_all", pd.DataFrame())
    if len(summary) and len(valid) and "track_key" in valid:
        crops = valid.groupby("track_key", as_index=False).agg(crop_count=("crop_path", "count"))
        summary = summary.merge(crops, on="track_key", how="left")
    if result.get("track_summary_gt_df") is not None and len(result["track_summary_gt_df"]):
        gt_cols = ["track_key", "dominant_gt_id", "track_gt_purity"]
        existing = [col for col in gt_cols if col in result["track_summary_gt_df"].columns]
        summary = summary.merge(result["track_summary_gt_df"][existing], on="track_key", how="left")
        summary = summary.rename(columns={"track_gt_purity": "track_purity"})
    summary["status"] = "valid"
    return summary


def _ringkasan_track_df(result: dict) -> pd.DataFrame:
    summary = result.get("valid_summary_all", pd.DataFrame()).copy()
    if len(summary) and "is_valid" in summary.columns:
        rows = summary.copy()
        if "num_crops" not in rows.columns:
            rows["num_crops"] = rows.get("crop_count", 0)
        rows["num_crops"] = rows["num_crops"].fillna(0).astype(int)
        valid_mask = _truthy_series(rows["is_valid"])
        rows["is_valid"] = valid_mask
        rows["status"] = rows.get("status", valid_mask.map(lambda ok: "valid" if ok else "filtered"))
        track_gt = result.get("track_summary_gt_df", pd.DataFrame())
        if track_gt is not None and len(track_gt):
            gt_cols = [c for c in ["track_key", "dominant_gt_id", "track_gt_purity"] if c in track_gt.columns]
            if gt_cols:
                rows = rows.merge(track_gt[gt_cols], on="track_key", how="left")
                rows = rows.rename(columns={"track_gt_purity": "track_purity"})
        return rows.sort_values(["camera", "track_id"], kind="stable")

    raw = result.get("raw_tracks_all", pd.DataFrame()).copy()
    valid = result.get("valid_tracks_all", pd.DataFrame()).copy()
    if raw is None or len(raw) == 0:
        if len(summary):
            summary["is_valid"] = True
            summary["num_crops"] = summary.get("crop_count", summary.get("num_crops", 0))
        return summary
    if "track_key" not in raw and {"camera", "track_id"}.issubset(raw.columns):
        raw["track_key"] = raw["camera"].astype(str) + "_T" + raw["track_id"].astype(int).astype(str)
    valid_keys = set(valid["track_key"].astype(str)) if len(valid) and "track_key" in valid else set()
    rows = (
        raw.groupby(["camera", "track_id", "track_key"], as_index=False)
        .agg(
            num_frames=("frame", "nunique"),
            avg_conf=("conf", "mean"),
            avg_area=("area", "mean"),
        )
    )
    rows["is_valid"] = rows["track_key"].astype(str).isin(valid_keys)
    if len(valid) and "crop_path" in valid:
        crops = valid.groupby("track_key", as_index=False).agg(num_crops=("crop_path", "count"))
        rows = rows.merge(crops, on="track_key", how="left")
    rows["num_crops"] = rows.get("num_crops", 0).fillna(0).astype(int)
    rows["status"] = rows["is_valid"].map(lambda ok: "valid" if ok else "filtered")
    track_gt = result.get("track_summary_gt_df", pd.DataFrame())
    if track_gt is not None and len(track_gt):
        gt_cols = [c for c in ["track_key", "dominant_gt_id", "track_gt_purity"] if c in track_gt.columns]
        if gt_cols:
            rows = rows.merge(track_gt[gt_cols], on="track_key", how="left")
            rows = rows.rename(columns={"track_gt_purity": "track_purity"})
    return rows.sort_values(["camera", "track_id"], kind="stable")


def _valid_track_count(result: dict) -> int:
    summary = result.get("valid_summary_all", pd.DataFrame())
    if summary is not None and len(summary) and {"track_key", "is_valid"}.issubset(summary.columns):
        return int(summary[_truthy_series(summary["is_valid"])]["track_key"].nunique())
    valid = result.get("valid_tracks_all", pd.DataFrame())
    return int(valid["track_key"].nunique()) if valid is not None and len(valid) and "track_key" in valid else 0


def _tracking_metrics_df(result: dict) -> pd.DataFrame:
    eval_df = result.get("tracking_eval_df", pd.DataFrame())
    if eval_df is None or len(eval_df) == 0:
        return pd.DataFrame()
    row = eval_df.iloc[0].to_dict()
    coverage = result.get("gt_coverage_df", pd.DataFrame())
    track_gt = result.get("track_summary_gt_df", pd.DataFrame())
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


def _tracking_score_df(result: dict, metrics_df: pd.DataFrame) -> pd.DataFrame:
    valid_track_count = _valid_track_count(result)
    if metrics_df is None or len(metrics_df) == 0 or not result.get("gt_available"):
        return pd.DataFrame([{
            "score_available": False,
            "tracking_score": None,
            "reason": "GT tidak tersedia; hanya operational summary yang ditampilkan.",
            "num_valid_tracks": valid_track_count,
            "runtime_sec": round(float(result.get("runtime_sec", 0)), 2),
        }])

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
        "runtime_sec": round(float(result.get("runtime_sec", 0)), 2),
    }])


def _gt_coverage_table(result: dict) -> pd.DataFrame:
    df = result.get("gt_coverage_df", pd.DataFrame())
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.rename(columns={
        "gt_frames": "gt_total_rows",
        "detected_frames": "matched_gt_rows",
        "coverage_per_gt": "gt_coverage_rate",
    }).copy()
    matched = result.get("matched_df", pd.DataFrame())
    if matched is not None and len(matched) and "gt_id" in matched and "track_key" in matched:
        matched_gt = matched[matched["gt_id"] > 0].copy()
        track_lists = (
            matched_gt.groupby(["camera", "gt_id"])["track_key"]
            .agg(lambda values: ", ".join(sorted(set(map(str, values)))))
            .reset_index(name="dominant_tracks")
        )
        out = out.merge(track_lists, on=["camera", "gt_id"], how="left")
    else:
        out["dominant_tracks"] = ""
    out["status"] = out["gt_coverage_rate"].apply(lambda x: "covered" if float(x) > 0 else "missed")
    return out


def _pairwise_confusion_matrix_df(pairwise_eval_df: pd.DataFrame | None) -> pd.DataFrame:
    if pairwise_eval_df is None or len(pairwise_eval_df) == 0:
        return pd.DataFrame()
    row = pairwise_eval_df.iloc[0].to_dict()
    return pd.DataFrame([
        {
            "GT / Prediction": "Same GT",
            "Same Global ID": int(row.get("tp_pair", 0) or 0),
            "Different Global ID": int(row.get("fn_pair", 0) or 0),
        },
        {
            "GT / Prediction": "Different GT",
            "Same Global ID": int(row.get("fp_pair", 0) or 0),
            "Different Global ID": int(row.get("tn_pair", 0) or 0),
        },
    ])


def _visual_preset_label_map() -> dict:
    return {
        f"{preset['label']} ({key})": key
        for key, preset in VISUAL_PRESETS.items()
    }


def _apply_visual_preset_to_case(cfg: dict, visual_key: str) -> None:
    preset = VISUAL_PRESETS[visual_key]
    cfg["tuning_preset"] = visual_key
    cfg["tracking"] = copy.deepcopy(preset["tracking"])
    cfg["filter"] = copy.deepcopy(preset["filter"])


def _apply_visual_preset_to_camera(cfg: dict, camera: str, visual_key: str) -> None:
    preset = VISUAL_PRESETS[visual_key]
    cfg.setdefault("camera_configs", {}).setdefault(camera, {})
    cfg["camera_configs"][camera]["tuning_preset"] = visual_key
    cfg["camera_configs"][camera]["profile"] = visual_key
    cfg["camera_configs"][camera]["tracking"] = copy.deepcopy(preset["tracking"])
    cfg["camera_configs"][camera]["filter"] = copy.deepcopy(preset["filter"])


def _default_profile_for_case(case: dict, camera: str) -> str:
    scene = str(case.get("scene_type", "")).lower()
    if "small" in scene or "distant" in scene:
        return "distant_small"
    if "crowded" in scene or "stress" in scene:
        return "crowded_occlusion"
    return "clear_large"


def _ensure_camera_configs(case: dict, cfg: dict) -> dict:
    cfg = normalize_config(cfg)
    camera_configs = cfg.setdefault("camera_configs", {})
    recommended_camera_configs = get_case_camera_configs(case["case_id"])
    for camera in case.get("cameras", []):
        if camera not in camera_configs:
            if camera in recommended_camera_configs:
                camera_configs[camera] = copy.deepcopy(recommended_camera_configs[camera])
            else:
                camera_configs[camera] = {
                    "profile": f"manual_{camera}",
                    "tracking": copy.deepcopy(cfg["tracking"]),
                    "filter": copy.deepcopy(cfg["filter"]),
                }
        else:
            camera_configs[camera].setdefault("profile", f"manual_{camera}")
            camera_configs[camera].setdefault("tracking", {})
            camera_configs[camera].setdefault("filter", {})
    cfg["camera_configs"] = camera_configs
    cfg["tracking_config_mode"] = "per_camera"
    return cfg


def multi_camera_tracking_config_panel(case: dict, state: dict) -> dict:
    case_id = case["case_id"]
    _init_config_state(case, state)
    cfg = _ensure_camera_configs(case, state["config"])
    if len(case.get("cameras", [])) > 3:
        st.error("Multi-camera flow ini mendukung maksimal 3 kamera/video.")
        st.stop()

    st.caption("Setiap kamera memakai tracking dan filter config sendiri. Jika nilai kamera diubah, status kamera otomatis menjadi konfigurasi manual dan backend langsung memakai nilai baru.")
    cfg.setdefault("custom", {"tracking": False, "filter": False, "reid": False})

    for camera in case["cameras"]:
        with st.expander(f"{camera} - konfigurasi tracking dan filter", expanded=True):
            cam_cfg = cfg["camera_configs"][camera]
            source = cam_cfg.get("tracking_config_source") or "Rekomendasi case"
            selected_source = st.selectbox(
                "Sumber konfigurasi tracking",
                TRACKING_SOURCE_OPTIONS,
                index=TRACKING_SOURCE_OPTIONS.index(source) if source in TRACKING_SOURCE_OPTIONS else 0,
                key=_safe_key(case_id, camera, "tracking_config_source_select"),
            )

            tracking_scope = _safe_key(case_id, camera, "tracking")
            filter_scope = _safe_key(case_id, camera, "filter")

            if selected_source != source:
                _apply_tracking_source_to_camera(cfg, case_id, camera, selected_source)
                cam_cfg = cfg["camera_configs"][camera]
                cam_cfg["tracking_config_dirty"] = False
                cam_cfg["filter_config_dirty"] = False
                _prime_tracking_widgets(tracking_scope, cam_cfg["tracking"], force=True)
                _prime_filter_widgets(filter_scope, cam_cfg["filter"], force=True)
                state["config"] = normalize_config(cfg)
                state["runtime_config"] = normalize_config(cfg)
                _invalidate_tracking_state(state)
                st.rerun()

            cam_cfg = cfg["camera_configs"][camera]
            before_tracking = copy.deepcopy(cam_cfg.get("tracking", {}))
            before_filter = copy.deepcopy(cam_cfg.get("filter", {}))

            left, right = st.columns(2)
            with left:
                st.markdown("#### Parameter tracking")
                cam_cfg["tracking"] = _tracking_custom_inputs(tracking_scope, cam_cfg.get("tracking") or cfg["tracking"])
            with right:
                st.markdown("#### Parameter filter")
                cam_cfg["filter"] = _filter_custom_inputs(filter_scope, cam_cfg.get("filter") or cfg["filter"])

            is_clean = _tracking_filter_equal_to_source(case_id, selected_source, cam_cfg["tracking"], cam_cfg["filter"], camera)
            cam_cfg["tracking_config_source"] = selected_source
            cam_cfg["tracking_config_dirty"] = not is_clean
            cam_cfg["filter_config_dirty"] = not is_clean
            _status_badge(_source_status(selected_source, is_clean))

            if selected_source != "Rekomendasi case":
                preset_key = TRACKING_SOURCE_TO_PRESET[selected_source]
                st.caption(VISUAL_PRESETS[preset_key]["description"])
            elif cam_cfg.get("profile"):
                st.caption(f"Profil rekomendasi: {cam_cfg.get('profile')}")

            if before_tracking != cam_cfg["tracking"] or before_filter != cam_cfg["filter"]:
                _invalidate_tracking_state(state)

    state["config"] = normalize_config(cfg)
    state["runtime_config"] = normalize_config(state["config"])
    return state["config"]


def _camera_valid_ready(case: dict, run_dir: str | Path | None) -> dict:
    if not run_dir:
        return {camera: False for camera in case.get("cameras", [])}
    run_dir = Path(run_dir)
    return {
        camera: (run_dir / camera / "local_tracks_valid.csv").exists()
        for camera in case.get("cameras", [])
    }


def _all_cameras_ready(case: dict, run_dir: str | Path | None) -> bool:
    ready = _camera_valid_ready(case, run_dir)
    return bool(ready) and all(ready.values()) and bool(run_dir) and (Path(run_dir) / "valid_tracks_all.csv").exists()


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _show_local_gallery(gallery_df: pd.DataFrame | None, title: str) -> None:
    st.markdown(f"#### {title}")
    if gallery_df is None or len(gallery_df) == 0:
        st.info("Gallery belum tersedia.")
        return
    cols = st.columns(4)
    for i, (_, row) in enumerate(gallery_df.iterrows()):
        with cols[i % 4]:
            path = Path(str(row.get("image_path", "")))
            if path.exists():
                st.image(str(path), use_container_width=True)
            lines = [
                str(row.get("track_key", "")),
                f"camera: {row.get('camera', '-')}",
                f"track_id: {row.get('track_id', '-')}",
                f"frames: {row.get('num_frames', '-')}",
                f"conf: {float(row.get('avg_conf', 0) or 0):.3f}",
                f"area: {float(row.get('avg_area', 0) or 0):.0f}",
            ]
            if pd.notna(row.get("dominant_gt_id", None)):
                lines.append(f"GT: {row.get('dominant_gt_id')}")
            if pd.notna(row.get("track_purity", None)):
                lines.append(f"purity: {float(row.get('track_purity') or 0):.3f}")
            if row.get("profile"):
                lines.append(f"profile: {row.get('profile')}")
            st.caption(" | ".join(lines))


def _show_global_gallery_filtered(result: dict, case_id: str) -> None:
    gallery_df = result.get("global_gallery_df", pd.DataFrame())
    meta = result.get("global_meta_df", pd.DataFrame())
    if gallery_df is None or len(gallery_df) == 0:
        show_gallery(gallery_df, "Galeri Global ID")
        return
    merged = gallery_df.copy().sort_values("global_id", kind="stable")
    if meta is not None and len(meta) and "global_id" in meta:
        rows = []
        summary = _global_summary_table(result)
        summary_map = {
            int(r["global_id"]): r.to_dict()
            for _, r in summary.iterrows()
        } if len(summary) else {}
        for _, row in merged.iterrows():
            gid = int(row["global_id"])
            info = summary_map.get(gid, {})
            out = row.to_dict()
            out.update({
                "members": info.get("members", row.get("tracks", "")),
                "cameras": info.get("cameras", row.get("cameras", "")),
                "num_tracks": info.get("num_tracks", row.get("num_tracks", 0)),
                "dominant_gt_id": info.get("dominant_gt_id"),
                "global_id_purity": info.get("global_id_purity"),
                "is_mixed_gid": info.get("is_mixed_gid", False),
            })
            rows.append(out)
        merged = pd.DataFrame(rows)

    mode = st.selectbox(
        "Filter Galeri Global ID",
        ["Semua Global ID", "Hanya multi-camera", "Hanya single-camera", "Hanya mixed GID"],
        key=_safe_key(case_id, "global_gallery_filter"),
    )
    shown = merged
    if mode == "Hanya multi-camera" and len(shown):
        shown = shown[shown["cameras"].astype(str).apply(lambda text: len([x for x in text.split(",") if x.strip()]) > 1)]
    elif mode == "Hanya single-camera" and len(shown):
        shown = shown[shown["cameras"].astype(str).apply(lambda text: len([x for x in text.split(",") if x.strip()]) <= 1)]
    elif mode == "Hanya mixed GID" and len(shown) and "is_mixed_gid" in shown:
        shown = shown[shown["is_mixed_gid"] == True]

    st.subheader("Galeri Global ID")
    if len(shown) == 0:
        st.info("Tidak ada Global ID untuk filter ini.")
        return
    cols = st.columns(3)
    for i, (_, row) in enumerate(shown.iterrows()):
        with cols[i % 3]:
            gid = int(row.get("global_id", 0))
            members = []
            try:
                members = json.loads(row.get("members_json", "[]"))
            except Exception:
                members = []
            st.markdown(f"#### Global ID {gid}")
            st.caption(f"Kamera: {row.get('cameras', '')} | {int(row.get('num_tracks', 0) or 0)} track")
            preview_members = members[:4]
            thumb_cols = st.columns(2 if len(preview_members) > 1 else 1)
            for j, member in enumerate(preview_members):
                with thumb_cols[j % len(thumb_cols)]:
                    p = Path(str(member.get("thumbnail_path", "")))
                    if p.exists():
                        st.image(str(p), use_container_width=True)
                    st.caption(f"{member.get('camera', '-')} T{member.get('track_id', '-')}")
            remaining = max(0, len(members) - 4)
            if remaining:
                st.caption(f"+{remaining} track lainnya")
            if pd.notna(row.get("dominant_gt_id", None)):
                st.caption(f"Dominant GT: {row.get('dominant_gt_id')}")
            if pd.notna(row.get("global_id_purity", None)):
                st.caption(f"Purity: {float(row.get('global_id_purity') or 0):.3f}")
            with st.expander("Lihat anggota", expanded=False):
                if members:
                    st.dataframe(pd.DataFrame(members), use_container_width=True, hide_index=True)
                else:
                    st.caption(row.get("members", row.get("tracks", "")))


def _show_merge_gallery(result: dict, case_id: str) -> None:
    run_dir = Path(result.get("run_dir", "")) if result.get("run_dir") else None
    gallery_df = result.get("merge_gallery_df", pd.DataFrame())
    if (gallery_df is None or len(gallery_df) == 0) and run_dir:
        gallery_df = _safe_read_csv(run_dir / "merge_gallery" / "merge_gallery_index.csv")
    shown = gallery_df.copy() if gallery_df is not None else pd.DataFrame()

    st.markdown("#### Bukti merge track")
    if shown is None or len(shown) == 0:
        st.info("Belum ada pasangan track yang digabung.")
    else:
        shown = shown.sort_values(["global_id", "similarity"], ascending=[True, False], kind="stable")
        cols = st.columns(3)
        for i, (_, row) in enumerate(shown.iterrows()):
            with cols[i % 3]:
                path = Path(str(row.get("image_path", "")))
                if path.exists():
                    st.image(str(path), use_container_width=True)
                sim = row.get("similarity", None)
                sim_text = "-" if pd.isna(sim) else f"{float(sim):.3f}"
                st.caption(
                    " | ".join([
                        f"GID {row.get('global_id', '-')}",
                        str(row.get("merge_type", "-")),
                        f"{row.get('camera_a', '-')} T{row.get('track_id_a', '-')} <-> {row.get('camera_b', '-')} T{row.get('track_id_b', '-')}",
                        f"sim={sim_text}",
                    ])
                )


def _load_camera_result_from_disk(run_dir: str | Path, camera: str) -> dict:
    cam_dir = Path(run_dir) / camera
    return {
        "camera": camera,
        "run_dir": cam_dir,
        "valid_summary": _safe_read_csv(cam_dir / "valid_track_summary.csv"),
        "gt_coverage_df": _safe_read_csv(cam_dir / "gt_coverage_detail.csv"),
        "tracking_score_df": _safe_read_csv(cam_dir / "flow1_tracking_score.csv"),
        "tracking_standard_metrics_df": _safe_read_csv(cam_dir / "tracking_standard_metrics.csv"),
        "local_gallery_df": _safe_read_csv(cam_dir / "gallery_local" / "local_gallery_index.csv"),
    }


def multi_camera_tracking_panel(case: dict, state: dict, use_cuda: bool, yolo_weight: str | None) -> None:
    case_id = case["case_id"]
    st.markdown("### Tahap 1 - Tracking lokal")
    cfg = multi_camera_tracking_config_panel(case, state)
    state["runtime_config"] = normalize_config(cfg)

    run_dir = state.get("run_dir")
    if run_dir is None:
        run_dir = None

    status_df = state.get("camera_status_df")
    if status_df is None and run_dir and (Path(run_dir) / "camera_tracking_status.csv").exists():
        status_df = _safe_read_csv(Path(run_dir) / "camera_tracking_status.csv")
        state["camera_status_df"] = status_df
    status_map = dict(zip(status_df["camera"], status_df["status"])) if status_df is not None and len(status_df) else {}

    st.markdown("#### Workspace kamera")
    camera_cols = st.columns(min(3, len(case["cameras"])))
    camera_to_run = None
    for idx, camera in enumerate(case["cameras"]):
        with camera_cols[idx % len(camera_cols)]:
            video_path = Path(case["video_files"][camera])
            frame = read_frame(video_path, 0)
            if frame is not None:
                st.image(frame, channels="BGR", use_container_width=True)
            info = get_video_info(video_path)
            tracking_cfg, filter_cfg = get_camera_stage_config(state["runtime_config"], camera)
            profile = ((state["runtime_config"].get("camera_configs") or {}).get(camera, {}) or {}).get("profile")
            status = status_map.get(camera, "not_run")
            st.markdown(f"**{camera}**")
            st.caption(f"video: {'ada' if video_path.exists() else 'tidak ada'} | status: {status}")
            if info.get("exists"):
                st.caption(f"{int(info.get('frames', 0))} frame | {float(info.get('duration_sec', 0)):.1f}s | {info.get('width', 0)}x{info.get('height', 0)}")
            st.caption(f"profile: {profile or '-'}")
            if st.button("Edit config", key=_safe_key(case_id, camera, "edit_config_hint")):
                st.info("Ubah konfigurasi kamera ini dari expander Tracking + Filter Config di atas.")
            if st.button("Jalankan kamera ini", key=_safe_key(case_id, camera, "run_this_camera")):
                camera_to_run = camera
            if run_dir and (Path(run_dir) / camera / "flow1_tracking_score.csv").exists():
                score_df = _safe_read_csv(Path(run_dir) / camera / "flow1_tracking_score.csv")
                if len(score_df) and pd.notna(score_df.iloc[0].get("tracking_score")):
                    st.metric("Score", f"{float(score_df.iloc[0]['tracking_score']):.3f}")
                valid_path = Path(run_dir) / camera / "local_tracks_valid.csv"
                if valid_path.exists():
                    valid_df = _safe_read_csv(valid_path)
                    st.metric("Valid tracks", valid_df["track_key"].nunique() if len(valid_df) and "track_key" in valid_df else 0)

    run_all = st.button("Jalankan tracking semua kamera", type="primary", key=_safe_key(case_id, "run_tracking_all"))
    if run_all or camera_to_run:
        cameras = case["cameras"] if run_all else [camera_to_run]
        progress = st.progress(0)
        status = st.empty()

        def on_progress(camera: str, done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            progress.progress(min(100, pct))
            status.write(f"Tracking {camera}: {done}/{total} frame ({pct}%)")

        try:
            start = time.perf_counter()
            result = run_tracking_stage(
                case=case,
                config=state["runtime_config"],
                yolo_weight=yolo_weight,
                use_cuda=use_cuda,
                run_dir=state.get("run_dir"),
                progress_callback=on_progress,
                cameras_to_run=list(cameras),
            )
            result["runtime_sec"] = time.perf_counter() - start
            state["tracking_result"] = result
            state["camera_status_df"] = result.get("camera_status_df")
            state["run_dir"] = str(result["run_dir"])
            state["tracking_done"] = _all_cameras_ready(case, state["run_dir"])
            state["reid_done"] = False
            state["render_done"] = False
            state["reid_result"] = None
            progress.progress(100)
            status.success("Tracking kamera selesai dan output gabungan diperbarui.")
        except Exception as e:
            st.error(f"Tracking gagal: {e}")
            st.exception(e)

    run_dir = state.get("run_dir")
    if not run_dir:
        return

    ready_map = _camera_valid_ready(case, run_dir)
    state["tracking_done"] = _all_cameras_ready(case, run_dir)
    if not any(ready_map.values()):
        return

    if not state.get("tracking_result") and (Path(run_dir) / "valid_tracks_all.csv").exists():
        state["tracking_result"] = {
            "run_dir": Path(run_dir),
            "raw_tracks_all": _safe_read_csv(Path(run_dir) / "local_tracks.csv"),
            "valid_tracks_all": _safe_read_csv(Path(run_dir) / "valid_tracks_all.csv"),
            "valid_summary_all": _safe_read_csv(Path(run_dir) / "valid_track_summary.csv"),
            "local_gallery_df": _safe_read_csv(Path(run_dir) / "gallery_local" / "local_gallery_index.csv"),
            "tracking_score_by_camera_df": _safe_read_csv(Path(run_dir) / "tracking_score_by_camera.csv"),
            "tracking_standard_metrics_df": _safe_read_csv(Path(run_dir) / "tracking_standard_metrics.csv"),
            "tracking_score_df": _safe_read_csv(Path(run_dir) / "flow1_tracking_score.csv"),
        }

    result = state.get("tracking_result") or {}
    tabs = st.tabs(["Ringkasan tracking", "Hasil per kamera", "Galeri local track", "Cakupan Ground Truth", "Diagnostik", "Unduhan"])

    with tabs[0]:
        st.markdown("#### Ringkasan tracking")
        valid = result.get("valid_tracks_all", pd.DataFrame())
        score_df = result.get("tracking_score_df", pd.DataFrame())
        if score_df is None or len(score_df) == 0:
            score_df = _safe_read_csv(Path(run_dir) / "flow1_tracking_score.csv")
        standard = result.get("tracking_standard_metrics_df", pd.DataFrame())
        if standard is None or len(standard) == 0:
            standard = _safe_read_csv(Path(run_dir) / "tracking_standard_metrics.csv")
        cols = st.columns(6)
        if len(score_df) and pd.notna(score_df.iloc[0].get("tracking_score")):
            cols[0].metric("Tracking score", f"{float(score_df.iloc[0]['tracking_score']):.3f}")
            if pd.notna(score_df.iloc[0].get("camera_score_mean", None)):
                cols[1].metric("Camera mean", f"{float(score_df.iloc[0]['camera_score_mean']):.3f}")
            if pd.notna(score_df.iloc[0].get("camera_score_min", None)):
                cols[2].metric("Camera min", f"{float(score_df.iloc[0]['camera_score_min']):.3f}")
        cols[3].metric("Valid tracks", _valid_track_count(result))
        if len(standard):
            row = standard.iloc[0].to_dict()
            cols[4].metric("F1", f"{float(row.get('f1', 0) or 0):.3f}")
            cols[5].metric("MOTA", f"{float(row.get('mota_simple', 0) or 0):.3f}")
            metric_cols = st.columns(5)
            metric_cols[0].metric("Precision", f"{float(row.get('precision', 0) or 0):.3f}")
            metric_cols[1].metric("Recall", f"{float(row.get('recall', 0) or 0):.3f}")
            metric_cols[2].metric("FP", int(row.get("fp", 0) or 0))
            metric_cols[3].metric("FN", int(row.get("fn", 0) or 0))
            metric_cols[4].metric("IDSW", int(row.get("id_switch_count", 0) or 0))
        score_by_camera = result.get("tracking_score_by_camera_df", pd.DataFrame())
        if score_by_camera is None or len(score_by_camera) == 0:
            score_by_camera = _safe_read_csv(Path(run_dir) / "tracking_score_by_camera.csv")
        standard_by_camera = _safe_read_csv(Path(run_dir) / "tracking_standard_metrics_by_camera.csv")
        display_score = score_by_camera.copy()
        if len(display_score) and len(standard_by_camera) and "camera" in standard_by_camera:
            mota_cols = [col for col in ["camera", "mota_simple"] if col in standard_by_camera.columns]
            display_score = display_score.merge(
                standard_by_camera[mota_cols].rename(columns={"mota_simple": "MOTA"}),
                on="camera",
                how="left",
            )
        _show_table(_ringkasan_track_df(result), "Ringkasan Track", "ringkasan_track.csv", _safe_key(case_id, "download", "ringkasan_track"), ["camera", "track_id", "track_key", "num_frames", "num_crops", "avg_conf", "avg_area", "min_frames_used", "min_crops_used", "min_avg_conf_used", "min_avg_area_used", "pass_min_frames", "pass_min_crops", "pass_min_avg_conf", "pass_min_avg_area", "is_valid", "status", "filter_reason", "dominant_gt_id", "track_purity"], run_dir=run_dir)
        with st.expander("Skor per kamera", expanded=False):
            _show_table(
                display_score,
                "Skor tracking per kamera",
                "tracking_score_by_camera.csv",
                _safe_key(case_id, "download", "score_by_camera"),
                ["camera", "tracking_score", "precision", "recall", "f1", "MOTA", "num_valid_tracks", "mixed_track_count", "runtime_sec"],
                run_dir=run_dir,
            )
        st.info("Re-ID baru aktif setelah semua kamera memiliki local_tracks_valid.csv dan valid_tracks_all.csv sudah siap.")

    with tabs[1]:
        cam_tabs = st.tabs(case["cameras"])
        for tab, camera in zip(cam_tabs, case["cameras"]):
            with tab:
                cam_result = (result.get("camera_results") or {}).get(camera)
                if cam_result is None:
                    cam_result = _load_camera_result_from_disk(run_dir, camera)
                score = cam_result.get("tracking_score_df", pd.DataFrame())
                standard = cam_result.get("tracking_standard_metrics_df", pd.DataFrame())
                cols = st.columns(5)
                if len(score) and pd.notna(score.iloc[0].get("tracking_score")):
                    cols[0].metric("Score", f"{float(score.iloc[0]['tracking_score']):.3f}")
                if len(standard):
                    row = standard.iloc[0].to_dict()
                    cols[1].metric("MOTA", f"{float(row.get('mota_simple', 0) or 0):.3f}")
                    cols[2].metric("Precision", f"{float(row.get('precision', 0) or 0):.3f}")
                    cols[3].metric("Recall", f"{float(row.get('recall', 0) or 0):.3f}")
                    cols[4].metric("F1", f"{float(row.get('f1', 0) or 0):.3f}")
                    sub_cols = st.columns(4)
                    sub_cols[0].metric("Fragmentation", f"{float(row.get('fragmentation_avg', 0) or 0):.2f}")
                    sub_cols[1].metric("FP rate", f"{float(row.get('false_positive_rate', 0) or 0):.3f}")
                    sub_cols[2].metric("ID Switch", int(row.get("id_switch_count", 0) or 0))
                    sub_cols[3].metric("Valid tracks", int(score.iloc[0].get("num_valid_tracks", 0) if len(score) else 0))
                _show_table(cam_result.get("valid_summary"), f"{camera} Ringkasan Track valid", f"{camera}_valid_track_summary.csv", _safe_key(case_id, camera, "valid_summary"))
                _show_table(cam_result.get("gt_coverage_df"), f"{camera} Cakupan Ground Truth", f"{camera}_gt_coverage_detail.csv", _safe_key(case_id, camera, "gt_coverage"))
                _show_local_gallery(cam_result.get("local_gallery_df"), f"{camera} Galeri local track")

    with tabs[2]:
        gallery = result.get("local_gallery_df", pd.DataFrame())
        if gallery is None or len(gallery) == 0:
            gallery = _safe_read_csv(Path(run_dir) / "gallery_local" / "local_gallery_index.csv")
        choices = ["All cameras"] + case["cameras"]
        selected = st.selectbox("Filter galeri", choices, key=_safe_key(case_id, "local_gallery_filter"))
        shown = gallery if selected == "All cameras" or gallery is None or len(gallery) == 0 else gallery[gallery["camera"] == selected]
        _show_local_gallery(shown, "Galeri local track")

    with tabs[3]:
        coverage = _safe_read_csv(Path(run_dir) / "gt_coverage_by_camera.csv")
        if coverage is None or len(coverage) == 0:
            st.info("Ground Truth tidak tersedia untuk case ini.")
        else:
            _show_table(coverage, "Cakupan Ground Truth per kamera", "gt_coverage_by_camera.csv", _safe_key(case_id, "download", "gt_coverage_by_camera"), run_dir=run_dir)

    with tabs[4]:
        hints = _safe_read_csv(Path(run_dir) / "tracking_tuning_hints_by_camera.csv")
        _show_table(hints, "Diagnostik dan saran tuning", "tracking_tuning_hints_by_camera.csv", _safe_key(case_id, "download", "hints_by_camera"), run_dir=run_dir)
        diagnostics = _safe_read_csv(Path(run_dir) / "tracking_diagnostics.csv")
        _show_table(diagnostics, "Diagnostik tracking agregat", "tracking_diagnostics.csv", _safe_key(case_id, "download", "tracking_diag"), run_dir=run_dir)

    with tabs[5]:
        for filename in [
            "local_tracks.csv",
            "valid_tracks_all.csv",
            "valid_track_summary.csv",
            "flow1_tracking_score.csv",
            "tracking_score_by_camera.csv",
            "tracking_standard_metrics_by_camera.csv",
            "gt_coverage_by_camera.csv",
            "camera_tracking_status.csv",
            "tracking_config_per_camera.json",
        ]:
            path = Path(run_dir) / filename
            if path.exists():
                _download_file(path, f"Unduh {filename}", _safe_key(case_id, "download_file", filename))


def tracking_panel(case: dict, state: dict, use_cuda: bool, yolo_weight: str | None) -> None:
    case_id = case["case_id"]
    if len(case.get("cameras", [])) > 1:
        multi_camera_tracking_panel(case, state, use_cuda, yolo_weight)
        return

    st.markdown("### Tahap 1 - Tracking lokal")
    cfg = tracking_config_panel(case, state)
    state["runtime_config"] = normalize_config(cfg)

    run = st.button("Jalankan tracking", type="primary", key=_safe_key(case_id, "run_tracking"))
    if run:
        progress = st.progress(0)
        status = st.empty()

        def on_progress(camera: str, done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            progress.progress(min(100, pct))
            status.write(f"Tracking {camera}: {done}/{total} frame ({pct}%)")

        try:
            start = time.perf_counter()
            result = run_tracking_stage(
                case=case,
                config=state["runtime_config"],
                yolo_weight=yolo_weight,
                use_cuda=use_cuda,
                progress_callback=on_progress,
            )
            runtime = time.perf_counter() - start
            result["runtime_sec"] = runtime
            state["tracking_done"] = True
            state["reid_done"] = False
            state["render_done"] = False
            state["tracking_result"] = result
            state["reid_result"] = None
            state["run_dir"] = str(result["run_dir"])
            progress.progress(100)
            status.success("Tracking selesai.")
        except Exception as e:
            st.error(f"Tracking gagal: {e}")
            st.exception(e)

    if not (state.get("tracking_done") and state.get("tracking_result")):
        return

    result = state["tracking_result"]
    raw = result.get("raw_tracks_all", pd.DataFrame())
    valid = result.get("valid_tracks_all", pd.DataFrame())
    summary = _tracking_summary_df(result, result.get("runtime_sec"))
    metrics_df = _tracking_metrics_df(result)
    tracking_score = _tracking_score_df(result, metrics_df)
    state["tracking_score_df"] = tracking_score

    st.success(f"Tracking selesai. Output: {result['run_dir']}")
    tabs = st.tabs(["Ringkasan tracking", "Hasil per kamera", "Galeri local track", "Cakupan Ground Truth", "Diagnostik", "Unduhan"])

    with tabs[0]:
        standard = result.get("tracking_standard_metrics_df", pd.DataFrame())
        row = standard.iloc[0].to_dict() if standard is not None and len(standard) else {}
        cols = st.columns(7)
        if bool(tracking_score.iloc[0].get("score_available", False)):
            cols[0].metric("Tracking score", f"{float(tracking_score.iloc[0]['tracking_score']):.3f}")
        else:
            cols[0].metric("Tracking score", "N/A")
        cols[1].metric("MOTA", "N/A" if not row else f"{float(row.get('mota_simple', 0) or 0):.3f}")
        cols[2].metric("Precision", "N/A" if not row else f"{float(row.get('precision', 0) or 0):.3f}")
        cols[3].metric("Recall", "N/A" if not row else f"{float(row.get('recall', 0) or 0):.3f}")
        cols[4].metric("F1", "N/A" if not row else f"{float(row.get('f1', 0) or 0):.3f}")
        cols[5].metric("Local track", raw["track_key"].nunique() if len(raw) and "track_key" in raw else 0)
        cols[6].metric("Valid track", _valid_track_count(result))
        st.metric("Jumlah kamera selesai", 1)
        if not bool(tracking_score.iloc[0].get("score_available", False)):
            st.info(str(tracking_score.iloc[0].get("reason", "Tracking score tidak tersedia.")))
        _show_table(
            _ringkasan_track_df(result),
            "Ringkasan Track",
            f"{case_id}_ringkasan_track.csv",
            _safe_key(case_id, "download", "ringkasan_track_tabs"),
            ["camera", "track_id", "track_key", "num_frames", "num_crops", "avg_conf", "avg_area", "min_frames_used", "min_crops_used", "min_avg_conf_used", "min_avg_area_used", "pass_min_frames", "pass_min_crops", "pass_min_avg_conf", "pass_min_avg_area", "is_valid", "status", "filter_reason", "dominant_gt_id", "track_purity"],
            run_dir=result["run_dir"],
        )

    with tabs[1]:
        camera = case.get("cameras", ["camera_1"])[0]
        standard = result.get("tracking_standard_metrics_df", pd.DataFrame())
        row = standard.iloc[0].to_dict() if standard is not None and len(standard) else {}
        cols = st.columns(5)
        cols[0].metric("Tracking score", "N/A" if not bool(tracking_score.iloc[0].get("score_available", False)) else f"{float(tracking_score.iloc[0]['tracking_score']):.3f}")
        cols[1].metric("MOTA", "N/A" if not row else f"{float(row.get('mota_simple', 0) or 0):.3f}")
        cols[2].metric("Precision", "N/A" if not row else f"{float(row.get('precision', 0) or 0):.3f}")
        cols[3].metric("Recall", "N/A" if not row else f"{float(row.get('recall', 0) or 0):.3f}")
        cols[4].metric("F1", "N/A" if not row else f"{float(row.get('f1', 0) or 0):.3f}")
        if row:
            sub_cols = st.columns(3)
            sub_cols[0].metric("Fragmentasi", f"{float(row.get('fragmentation_avg', 0) or 0):.2f}")
            sub_cols[1].metric("False positive rate", f"{float(row.get('false_positive_rate', 0) or 0):.3f}")
            sub_cols[2].metric("ID switch", int(row.get("id_switch_count", 0) or 0))
        _show_table(
            _ringkasan_track_df(result),
            f"{camera} Ringkasan Track",
            f"{camera}_ringkasan_track.csv",
            _safe_key(case_id, "download", "camera_ringkasan_track_tabs"),
            ["camera", "track_id", "track_key", "num_frames", "num_crops", "avg_conf", "avg_area", "min_frames_used", "min_crops_used", "min_avg_conf_used", "min_avg_area_used", "pass_min_frames", "pass_min_crops", "pass_min_avg_conf", "pass_min_avg_area", "is_valid", "status", "filter_reason", "dominant_gt_id", "track_purity"],
            run_dir=result["run_dir"],
        )
        _show_local_gallery(result.get("local_gallery_df"), f"{camera} Galeri local track")

    with tabs[2]:
        _show_local_gallery(result.get("local_gallery_df"), "Galeri local track")

    with tabs[3]:
        if result.get("gt_available"):
            _show_table(
                _gt_coverage_table(result),
                "Cakupan Ground Truth",
                f"{case_id}_gt_coverage.csv",
                _safe_key(case_id, "download", "gt_coverage_tabs"),
                ["gt_id", "gt_total_rows", "matched_gt_rows", "gt_coverage_rate", "dominant_tracks", "status"],
                run_dir=result["run_dir"],
            )
        else:
            st.info("Ground Truth tidak tersedia untuk case ini.")

    with tabs[4]:
        _show_table(
            result.get("tracking_diagnostics_df"),
            "Diagnostik tracking",
            f"{case_id}_tracking_diagnostics.csv",
            _safe_key(case_id, "download", "tracking_diagnostics_tabs"),
            run_dir=result["run_dir"],
        )
        hints_path = Path(result["run_dir"]) / "tracking_tuning_hints_by_camera.csv"
        hints = _safe_read_csv(hints_path) if hints_path.exists() else pd.DataFrame()
        _show_table(hints, "Saran tuning", "tracking_tuning_hints_by_camera.csv", _safe_key(case_id, "download", "tracking_hints_tabs"), run_dir=result["run_dir"])

    with tabs[5]:
        for filename in [
            "local_tracks.csv",
            "valid_tracks_all.csv",
            "valid_track_summary.csv",
            "ringkasan_track.csv",
            "flow1_tracking_score.csv",
            "tracking_standard_metrics.csv",
            "tracking_score_by_camera.csv",
            "gt_coverage_by_camera.csv",
            "tracking_config_per_camera.json",
        ]:
            path = Path(result["run_dir"]) / filename
            if path.exists():
                _download_file(path, f"Unduh {filename}", _safe_key(case_id, "download_file_single_tabs", filename))
    return

    st.success(f"Tracking selesai. Output: {result['run_dir']}")
    cols = st.columns(5)
    cols[0].metric("Raw detections", len(raw))
    cols[1].metric("Raw local tracks", raw["track_key"].nunique() if len(raw) and "track_key" in raw else 0)
    cols[2].metric("Valid tracks", _valid_track_count(result))
    cols[3].metric("Crops", valid["crop_path"].notna().sum() if len(valid) and "crop_path" in valid else len(valid))
    cols[4].metric("Runtime", f"{float(result.get('runtime_sec', 0)):.1f}s")

    if bool(tracking_score.iloc[0].get("score_available", False)):
        st.metric("Tracking Score", f"{float(tracking_score.iloc[0]['tracking_score']):.3f}")
    else:
        st.info(str(tracking_score.iloc[0].get("reason", "Tracking score tidak tersedia.")))

    _show_table(
        tracking_score,
        "Tracking Score",
        "flow1_tracking_score.csv",
        _safe_key(case_id, "download", "flow1_tracking_score"),
        run_dir=result["run_dir"],
    )
    _show_table(
        _ringkasan_track_df(result),
        "Ringkasan Track",
        f"{case_id}_ringkasan_track.csv",
        _safe_key(case_id, "download", "ringkasan_track"),
        ["camera", "track_id", "track_key", "num_frames", "num_crops", "avg_conf", "avg_area", "min_frames_used", "min_crops_used", "min_avg_conf_used", "min_avg_area_used", "pass_min_frames", "pass_min_crops", "pass_min_avg_conf", "pass_min_avg_area", "is_valid", "status", "filter_reason", "dominant_gt_id", "track_purity"],
        run_dir=result["run_dir"],
    )

    if result.get("gt_available"):
        _show_table(
            metrics_df,
            "Tracking Metrics",
            f"{case_id}_tracking_metrics.csv",
            _safe_key(case_id, "download", "tracking_metrics"),
            run_dir=result["run_dir"],
        )
        _show_table(
            _gt_coverage_table(result),
            "GT Coverage",
            f"{case_id}_gt_coverage.csv",
            _safe_key(case_id, "download", "gt_coverage"),
            ["gt_id", "gt_total_rows", "matched_gt_rows", "gt_coverage_rate", "dominant_tracks", "status"],
            run_dir=result["run_dir"],
        )
        st.info(
            "tracking_score adalah skor operasional untuk tuning konfigurasi. "
            "tracking_standard_metrics.csv adalah metrik akademik berbasis Ground Truth."
        )
        standard = result.get("tracking_standard_metrics_df", pd.DataFrame())
        if standard is not None and len(standard):
            row = standard.iloc[0].to_dict()
            metric_cols = st.columns(5)
            metric_cols[0].metric("TP", int(row.get("tp", 0) or 0))
            metric_cols[1].metric("FP", int(row.get("fp", 0) or 0))
            metric_cols[2].metric("FN", int(row.get("fn", 0) or 0))
            metric_cols[3].metric("F1", f"{float(row.get('f1', 0) or 0):.3f}")
            metric_cols[4].metric("MOTA simple", f"{float(row.get('mota_simple', 0) or 0):.3f}")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Precision", f"{float(row.get('precision', 0) or 0):.3f}")
            metric_cols[1].metric("Recall", f"{float(row.get('recall', 0) or 0):.3f}")
            metric_cols[2].metric("MOTP IoU", f"{float(row.get('motp_iou', 0) or 0):.3f}")
            metric_cols[3].metric("ID Switch", int(row.get("id_switch_count", 0) or 0))
        _show_table(
            standard,
            "Tracking Standard Metrics",
            "tracking_standard_metrics.csv",
            _safe_key(case_id, "download", "tracking_standard_metrics"),
            run_dir=result["run_dir"],
        )
    else:
        st.info("Ground truth tidak tersedia. Evaluasi tracking berbasis GT tidak aktif.")

    with st.expander("File detail tracking", expanded=False):
        download_table(result.get("valid_tracks_all"), "Unduh valid_tracks_all.csv", f"{case_id}_valid_tracks_all.csv", key=_safe_key(case_id, "download", "valid_tracks_all"))
        download_table(result.get("raw_tracks_all"), "Unduh local_tracks.csv", f"{case_id}_local_tracks.csv", key=_safe_key(case_id, "download", "local_tracks"))
        download_table(result.get("tracking_diagnostics_df"), "Unduh tracking_diagnostics.csv", f"{case_id}_tracking_diagnostics.csv", key=_safe_key(case_id, "download", "tracking_diagnostics"))
        download_table(result.get("gt_coverage_df"), "Unduh gt_coverage_detail.csv", f"{case_id}_gt_coverage_detail.csv", key=_safe_key(case_id, "download", "gt_coverage_detail"))
        download_table(result.get("track_summary_gt_df"), "Unduh track_summary_with_gt.csv", f"{case_id}_track_summary_with_gt.csv", key=_safe_key(case_id, "download", "track_summary_with_gt"))
        download_table(result.get("tracking_standard_metrics_df"), "Unduh tracking_standard_metrics.csv", f"{case_id}_tracking_standard_metrics.csv", key=_safe_key(case_id, "download", "tracking_standard_metrics_detail"))
        with st.expander("Cek sinkronisasi GT", expanded=False):
            st.json(result.get("gt_debug", {}))


def _global_summary_table(result: dict) -> pd.DataFrame:
    meta = result.get("global_meta_df", pd.DataFrame()).copy()
    if meta is None or len(meta) == 0:
        return pd.DataFrame()
    rows = []
    eval_df = result.get("global_eval_df", pd.DataFrame())
    eval_map = eval_df.set_index("global_id").to_dict("index") if eval_df is not None and len(eval_df) and "global_id" in eval_df else {}
    for gid, group in meta.groupby("global_id"):
        ev = eval_map.get(gid, {})
        gt_ids = str(ev.get("gt_ids_inside", "[]")).strip("[]")
        dominant_gt = gt_ids.split(",")[0].strip() if gt_ids else None
        rows.append({
            "global_id": int(gid),
            "num_tracks": int(group["track_key"].nunique()),
            "cameras": ", ".join(sorted(group["camera"].dropna().unique())),
            "first_frame": int(group["first_frame"].min()) if "first_frame" in group else None,
            "last_frame": int(group["last_frame"].max()) if "last_frame" in group else None,
            "members": ", ".join(sorted(group["track_key"].dropna().astype(str).unique())),
            "dominant_gt_id": dominant_gt,
            "global_id_purity": 1.0 if ev and bool(ev.get("is_pure_global_id")) else (0.0 if ev else None),
            "is_mixed_gid": (not bool(ev.get("is_pure_global_id"))) if ev else None,
        })
    return pd.DataFrame(rows).sort_values("global_id")


def _merged_pairs_table(pair_df: pd.DataFrame, global_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    if pair_df is None or len(pair_df) == 0 or "merge_status" not in pair_df:
        return pd.DataFrame()
    merged = pair_df[pair_df["merge_status"] == True].copy()
    if len(merged) == 0:
        return merged
    if "merge_type" not in merged and {"camera_a", "camera_b"}.issubset(merged.columns):
        merged["merge_type"] = merged.apply(
            lambda row: "intra-camera" if row.get("camera_a") == row.get("camera_b") else "cross-camera",
            axis=1,
        )
    if "global_id" not in merged and global_meta is not None and len(global_meta) and {"track_key", "global_id"}.issubset(global_meta.columns):
        gid_map = global_meta.drop_duplicates("track_key").set_index("track_key")["global_id"].to_dict()
        merged["global_id"] = merged["track_a"].map(gid_map) if "track_a" in merged else None
    if "cosine_similarity" in merged:
        sort_cols = [col for col in ["global_id", "cosine_similarity"] if col in merged.columns]
        if sort_cols:
            ascending = [True if col == "global_id" else False for col in sort_cols]
            merged = merged.sort_values(sort_cols, ascending=ascending)
    return merged


def _top_similarity_table(pair_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if pair_df is None or len(pair_df) == 0:
        return pd.DataFrame()
    return pair_df.sort_values("cosine_similarity", ascending=False).head(int(n)).copy()


def _reid_summary_metrics(result: dict, runtime: float | None) -> pd.DataFrame:
    meta = result.get("global_meta_df", pd.DataFrame())
    pairs = result.get("pair_df", pd.DataFrame())
    merged = _merged_pairs_table(pairs)
    local_tracks = int(meta["track_key"].nunique()) if len(meta) and "track_key" in meta else 0
    global_ids = int(meta["global_id"].nunique()) if len(meta) and "global_id" in meta else 0
    global_summary = result.get("global_summary_df", pd.DataFrame())
    pure_rate = None
    mixed_count = None
    global_score = None
    if global_summary is not None and len(global_summary):
        row = global_summary.iloc[0].to_dict()
        pure_rate = row.get("pure_global_id_rate")
        mixed_count = row.get("false_merge_count")
        global_score = row.get("score")
    return pd.DataFrame([{
        "num_local_tracks": local_tracks,
        "num_global_ids": global_ids,
        "num_merged_pairs": int(len(merged)),
        "fragment_reduction_rate": (local_tracks - global_ids) / max(1, local_tracks),
        "mean_global_id_purity": pure_rate,
        "mixed_gid_count": mixed_count,
        "global_id_purity": pure_rate,
        "runtime_reid_sec": round(float(runtime or 0), 2),
    }])


def _expected_global_ids_from_tracking(tracking_result: dict | None) -> int | None:
    if not tracking_result:
        return None
    matched = tracking_result.get("matched_df", pd.DataFrame())
    if matched is not None and len(matched) and "gt_id" in matched:
        represented = matched[matched["gt_id"] > 0]["gt_id"].dropna().astype(int).unique()
        return int(len(represented))
    track_gt = tracking_result.get("track_summary_gt_df", pd.DataFrame())
    if track_gt is not None and len(track_gt) and "dominant_gt_id" in track_gt:
        represented = track_gt[track_gt["dominant_gt_id"] > 0]["dominant_gt_id"].dropna().astype(int).unique()
        return int(len(represented))
    return None


def _reid_score_df(
    result: dict,
    metrics_df: pd.DataFrame,
    tracking_result: dict | None = None,
) -> pd.DataFrame:
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    row = metrics_df.iloc[0].to_dict()
    purity = row.get("mean_global_id_purity")
    mixed_count = row.get("mixed_gid_count")
    local_tracks = int(row.get("num_local_tracks", 0))
    global_ids = int(row.get("num_global_ids", 0))
    merged_pairs = int(row.get("num_merged_pairs", 0))
    expected_global_ids = _expected_global_ids_from_tracking(tracking_result)
    if pd.isna(purity) or mixed_count is None or pd.isna(mixed_count):
        return pd.DataFrame([{
            "score_available": False,
            "reid_score": None,
            "reason": "GT tidak tersedia; hanya operational summary yang ditampilkan.",
            "expected_global_ids": expected_global_ids,
            "reduction_required": None,
            "reduction_achieved": max(0, local_tracks - global_ids),
            "fragment_reduction_quality": None,
            "mean_global_id_purity": purity,
            "mixed_gid_quality": None,
            "merge_quality": None,
            "num_local_tracks": local_tracks,
            "num_global_ids": global_ids,
            "num_merged_pairs": merged_pairs,
            "mixed_gid_count": mixed_count,
            "runtime_reid_sec": row.get("runtime_reid_sec"),
        }])

    mixed_count = int(mixed_count)
    if expected_global_ids is None:
        expected_global_ids = global_ids
    reduction_required = max(0, local_tracks - int(expected_global_ids))
    reduction_achieved = max(0, local_tracks - global_ids)
    if reduction_required == 0:
        fragment_reduction_quality = 1.0
    else:
        fragment_reduction_quality = min(1.0, reduction_achieved / max(1, reduction_required))

    mixed_gid_quality = 1.0 - (mixed_count / max(1, global_ids))
    if reduction_required == 0 and mixed_count == 0:
        merge_quality = 1.0
    elif merged_pairs > 0:
        merge_quality = purity
    else:
        merge_quality = 0.0

    score_inputs = {
        "mean_global_id_purity": purity,
        "mixed_gid_quality": mixed_gid_quality,
        "fragment_reduction_quality": fragment_reduction_quality,
        "merge_quality": merge_quality,
    }
    score = _weighted_score(score_inputs, SCORE_WEIGHTS["reid"])
    return pd.DataFrame([{
        "score_available": True,
        "reid_score": score,
        "expected_global_ids": int(expected_global_ids),
        "reduction_required": int(reduction_required),
        "reduction_achieved": int(reduction_achieved),
        "fragment_reduction_quality": _clamp01(fragment_reduction_quality),
        "mean_global_id_purity": _clamp01(purity),
        "mixed_gid_quality": _clamp01(mixed_gid_quality),
        "merge_quality": _clamp01(merge_quality),
        "num_local_tracks": local_tracks,
        "num_global_ids": global_ids,
        "num_merged_pairs": merged_pairs,
        "mixed_gid_count": mixed_count,
        "runtime_reid_sec": row.get("runtime_reid_sec"),
    }])


def _final_score_df(tracking_score_df: pd.DataFrame, reid_score_df: pd.DataFrame) -> pd.DataFrame:
    if (
        tracking_score_df is None
        or reid_score_df is None
        or len(tracking_score_df) == 0
        or len(reid_score_df) == 0
        or not bool(tracking_score_df.iloc[0].get("score_available", False))
        or not bool(reid_score_df.iloc[0].get("score_available", False))
    ):
        return pd.DataFrame([{
            "score_available": False,
            "final_pipeline_score": None,
            "reason": "GT tidak tersedia atau salah satu tahap belum memiliki skor akurasi.",
        }])

    tracking_score = float(tracking_score_df.iloc[0]["tracking_score"])
    reid_score = float(reid_score_df.iloc[0]["reid_score"])
    final_score = _weighted_score(
        {
            "tracking_score": tracking_score,
            "reid_score": reid_score,
        },
        SCORE_WEIGHTS["final"],
    )
    return pd.DataFrame([{
        "score_available": True,
        "final_pipeline_score": final_score,
        "tracking_score": tracking_score,
        "reid_score": reid_score,
        "tracking_weight": SCORE_WEIGHTS["final"]["tracking_score"],
        "reid_weight": SCORE_WEIGHTS["final"]["reid_score"],
    }])


def reid_panel(case: dict, state: dict, use_cuda: bool, osnet_weight: str | None) -> None:
    case_id = case["case_id"]
    st.markdown("### Flow 2 - Re-ID / Global ID")
    is_multicamera = len(case.get("cameras", [])) > 1
    if is_multicamera:
        st.info(
            "Tracking config dapat berbeda per kamera, tetapi Re-ID config tetap case-level "
            "karena Global ID dibentuk dari seluruh track hasil semua kamera."
        )
        ready = _camera_valid_ready(case, state.get("run_dir"))
        valid_all_ready = bool(state.get("run_dir")) and (Path(state["run_dir"]) / "valid_tracks_all.csv").exists()
        checklist = pd.DataFrame([
            {"item": f"{camera} valid_tracks available", "ready": "yes" if ok else "no"}
            for camera, ok in ready.items()
        ] + [{"item": "valid_tracks_all.csv ready", "ready": "yes" if valid_all_ready else "no"}])
        st.markdown("#### Re-ID Readiness")
        st.dataframe(checklist, use_container_width=True, hide_index=True)
        state["tracking_done"] = bool(ready) and all(ready.values()) and valid_all_ready

    if not state.get("tracking_done"):
        st.info("Flow 2 terkunci. Jalankan Tracking Lokal untuk semua kamera sampai valid tracks tersedia.")
        return

    cfg = reid_config_panel(case, state)
    state["config"] = normalize_config(cfg)
    _set_runtime_config_from_state(state)
    run = st.button("Run Re-ID", type="primary", key=_safe_key(case_id, "run_reid"), disabled=not state.get("tracking_done"))

    if run:
        reid_progress = st.progress(0)
        reid_status = st.empty()

        def on_reid(done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            reid_progress.progress(min(100, pct))
            reid_status.write(f"Ekstraksi embedding: {done}/{total} crop ({pct}%)")

        try:
            start = time.perf_counter()
            result = run_reid_stage(
                case=case,
                config=state["runtime_config"],
                run_dir=state["run_dir"],
                osnet_weight=osnet_weight,
                use_cuda=use_cuda,
                reid_progress_callback=on_reid,
                render=False,
            )
            result["runtime_sec"] = time.perf_counter() - start
            state["reid_done"] = True
            state["render_done"] = False
            state["reid_result"] = result
            reid_progress.progress(100)
            reid_status.success("Re-ID selesai.")
            for message in result.get("stage_log", []):
                st.info(message)
        except Exception as e:
            st.error(f"Re-ID gagal: {e}")
            st.exception(e)

    if not (state.get("reid_done") and state.get("reid_result")):
        return

    result = state["reid_result"]
    if result.get("stage_log"):
        st.caption("Stage log: " + " | ".join(map(str, result.get("stage_log", []))))
    if result.get("reid_config_used"):
        with st.expander("Re-ID config used", expanded=False):
            st.json(result["reid_config_used"])
    metrics = _reid_summary_metrics(result, result.get("runtime_sec"))
    reid_score = _reid_score_df(result, metrics, state.get("tracking_result"))
    tracking_score = state.get("tracking_score_df")
    if tracking_score is None and state.get("tracking_result"):
        tracking_metrics = _tracking_metrics_df(state["tracking_result"])
        tracking_score = _tracking_score_df(state["tracking_result"], tracking_metrics)
        state["tracking_score_df"] = tracking_score
    final_score = _final_score_df(tracking_score, reid_score)
    state["reid_score_df"] = reid_score
    state["final_score_df"] = final_score
    row = metrics.iloc[0].to_dict()
    cols = st.columns(5)
    cols[0].metric("Local tracks", row["num_local_tracks"])
    cols[1].metric("Global IDs", row["num_global_ids"])
    cols[2].metric("Merged pairs", row["num_merged_pairs"])
    cols[3].metric("Fragment reduction", f"{row['fragment_reduction_rate']:.2%}")
    cols[4].metric("Runtime", f"{row['runtime_reid_sec']:.1f}s")

    if bool(reid_score.iloc[0].get("score_available", False)):
        st.metric("Re-ID / Global ID Score", f"{float(reid_score.iloc[0]['reid_score']):.3f}")
    else:
        st.info(str(reid_score.iloc[0].get("reason", "Re-ID score tidak tersedia.")))
    if bool(final_score.iloc[0].get("score_available", False)):
        st.metric("Final Pipeline Score", f"{float(final_score.iloc[0]['final_pipeline_score']):.3f}")
    else:
        st.info(str(final_score.iloc[0].get("reason", "Final score tidak tersedia.")))

    evidence_tab = st.tabs(["Merged Track Evidence"])[0]
    with evidence_tab:
        _show_merge_gallery(result, case_id)

    _show_table(
        reid_score,
        "Re-ID / Global ID Score",
        "flow2_global_id_score.csv",
        _safe_key(case_id, "download", "flow2_global_id_score"),
        run_dir=result["run_dir"],
    )
    _show_table(
        final_score,
        "Final Pipeline Score",
        "final_pipeline_score.csv",
        _safe_key(case_id, "download", "final_score"),
        run_dir=result["run_dir"],
    )
    _show_table(
        _global_summary_table(result),
        "Global ID Summary",
        f"{case_id}_global_id_summary.csv",
        _safe_key(case_id, "download", "global_id_summary"),
        ["global_id", "num_tracks", "members", "cameras", "first_frame", "last_frame", "dominant_gt_id", "global_id_purity", "is_mixed_gid"],
        run_dir=result["run_dir"],
    )
    _show_table(
        _merged_pairs_table(result.get("pair_df", pd.DataFrame())),
        "Merged Pairs",
        f"{case_id}_merged_pairs.csv",
        _safe_key(case_id, "download", "merged_pairs"),
        ["track_a", "track_b", "camera_a", "camera_b", "temporal_gap", "temporal_overlap", "cosine_similarity", "merge_reason"],
        run_dir=result["run_dir"],
    )
    top_n = st.slider("Top similarity candidates", 5, 100, 20, 5, key=_safe_key(case_id, "top_similarity_n"))
    _show_table(
        _top_similarity_table(result.get("pair_df", pd.DataFrame()), top_n),
        "Top Similarity Candidates",
        f"{case_id}_top_similarity_candidates.csv",
        _safe_key(case_id, "download", "top_similarity"),
        ["track_a", "track_b", "same_camera", "temporal_gap", "temporal_overlap", "cosine_similarity", "merge_status"],
        run_dir=result["run_dir"],
    )

    st.info(
        "reid_score dan final_pipeline_score adalah skor operasional untuk tuning konfigurasi. "
        "reid_pairwise_evaluation.csv adalah metrik akademik berbasis Ground Truth. "
        "Untuk case tanpa kebutuhan asosiasi, pairwise Re-ID metric ditampilkan sebagai N/A, bukan 0."
    )
    pairwise_eval = result.get("reid_pairwise_eval_df", pd.DataFrame())
    pairwise_detail = result.get("reid_pairwise_detail_df", pd.DataFrame())
    if pairwise_eval is not None and len(pairwise_eval):
        row = pairwise_eval.iloc[0].to_dict()
        st.markdown("#### Re-ID Pairwise Association Matrix")
        st.dataframe(_pairwise_confusion_matrix_df(pairwise_eval), use_container_width=True, hide_index=True)
        metric_cols = st.columns(5)
        assoc = bool(row.get("association_eval_available", False))
        metric_cols[0].metric("Association eval", "Available" if assoc else "N/A")
        metric_cols[1].metric("Pairwise Precision", "N/A" if pd.isna(row.get("pairwise_precision")) else f"{float(row.get('pairwise_precision')):.3f}")
        metric_cols[2].metric("Pairwise Recall", "N/A" if pd.isna(row.get("pairwise_recall")) else f"{float(row.get('pairwise_recall')):.3f}")
        metric_cols[3].metric("Pairwise F1", "N/A" if pd.isna(row.get("pairwise_f1")) else f"{float(row.get('pairwise_f1')):.3f}")
        metric_cols[4].metric("False Merge", "N/A" if pd.isna(row.get("false_merge_rate")) else f"{float(row.get('false_merge_rate')):.3f}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("False Split", "N/A" if pd.isna(row.get("false_split_rate")) else f"{float(row.get('false_split_rate')):.3f}")
        metric_cols[1].metric("Positive pairs", int(row.get("positive_pair_count", 0) or 0))
        metric_cols[2].metric("Eval tracks", int(row.get("num_eval_tracks", 0) or 0))
        st.caption(str(row.get("note", "")))
    _show_table(
        pairwise_eval,
        "Re-ID Pairwise Association Evaluation",
        "reid_pairwise_evaluation.csv",
        _safe_key(case_id, "download", "reid_pairwise_evaluation"),
        run_dir=result["run_dir"],
    )
    _show_table(
        pairwise_detail,
        "Re-ID Pairwise Detail",
        "reid_pairwise_detail.csv",
        _safe_key(case_id, "download", "reid_pairwise_detail"),
        [
            "track_a",
            "track_b",
            "camera_a",
            "camera_b",
            "global_id_a",
            "global_id_b",
            "dominant_gt_id_a",
            "dominant_gt_id_b",
            "track_purity_a",
            "track_purity_b",
            "same_gt",
            "same_global_id",
            "pair_eval_type",
            "cosine_similarity",
            "temporal_gap",
            "temporal_overlap",
            "merge_status",
            "merge_reason",
            "reject_reason",
        ],
        run_dir=result["run_dir"],
    )

    with st.expander("File detail Re-ID", expanded=False):
        download_table(result.get("pair_df"), "Unduh pair_similarity.csv", f"{case_id}_pair_similarity.csv", key=_safe_key(case_id, "download", "pair_similarity"))
        download_table(result.get("global_meta_df"), "Unduh global_track_meta.csv", f"{case_id}_global_track_meta.csv", key=_safe_key(case_id, "download", "global_track_meta"))
        download_table(result.get("reid_diagnostics_df"), "Unduh reid_diagnostics.csv", f"{case_id}_reid_diagnostics.csv", key=_safe_key(case_id, "download", "reid_diagnostics"))
        if result.get("global_eval_df") is not None and len(result["global_eval_df"]):
            download_table(result["global_eval_df"], "Unduh global_id_evaluation.csv", f"{case_id}_global_id_evaluation.csv", key=_safe_key(case_id, "download", "global_id_evaluation"))
            download_table(result["global_summary_df"], "Unduh global_id_summary_metrics.csv", f"{case_id}_global_id_summary_metrics.csv", key=_safe_key(case_id, "download", "global_id_summary_metrics"))
        download_table(result.get("reid_pairwise_eval_df"), "Unduh reid_pairwise_evaluation.csv", f"{case_id}_reid_pairwise_evaluation.csv", key=_safe_key(case_id, "download", "reid_pairwise_evaluation_detail"))
        download_table(result.get("reid_pairwise_detail_df"), "Unduh reid_pairwise_detail.csv", f"{case_id}_reid_pairwise_detail.csv", key=_safe_key(case_id, "download", "reid_pairwise_detail_detail"))


def render_output_panel(case: dict, state: dict) -> None:
    case_id = case["case_id"]
    st.markdown("### Render video")
    if not state.get("reid_done"):
        st.info("Render tersedia setelah Re-ID selesai.")
        return

    result = state["reid_result"]
    run_dir = Path(state["run_dir"])
    combined_layout = "fullscreen_grid"
    st.caption("Combined video memakai layout fullscreen 1920x1080.")
    if st.button("Render video", type="primary", key=_safe_key(case_id, "render_video")):
        progress = st.progress(0)
        status = st.empty()

        def on_render(camera: str, done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            progress.progress(min(100, pct))
            status.write(f"Render {camera}: {done}/{total} frame ({pct}%)")

        try:
            outputs = render_reid_outputs(
                case,
                run_dir,
                result.get("render_df"),
                on_render,
                combined_layout=combined_layout,
            )
            result.update(outputs)
            state["reid_result"] = result
            state["render_done"] = True
            progress.progress(100)
            status.success("Render selesai.")
        except Exception as e:
            st.error(f"Render gagal: {e}")
            st.exception(e)

    if result.get("combined_path") and Path(result["combined_path"]).exists():
        st.subheader("Combined video")
        st.video(str(result["combined_path"]))
    with st.expander("Video per kamera", expanded=False):
        for path in result.get("rendered_paths", []) or []:
            p = Path(path)
            if p.exists():
                st.markdown(f"#### {p.stem}")
                st.video(str(p))

    cols = st.columns(4)
    with cols[0]:
        _download_file(run_dir / "global_track_meta.csv", "Download global_track_meta.csv", _safe_key(case_id, "download", "global_meta_file"))
    with cols[1]:
        _download_file(run_dir / "merged_pairs.csv", "Download merged_pairs.csv", _safe_key(case_id, "download", "merged_pairs_file"))
    with cols[2]:
        _download_file(run_dir / "run_config.json", "Download run_config.json", _safe_key(case_id, "download", "run_config"))
    with cols[3]:
        if result.get("combined_path"):
            _download_file(result["combined_path"], "Download rendered video", _safe_key(case_id, "download", "rendered_video"))

    score_cols = st.columns(3)
    with score_cols[0]:
        _download_file(run_dir / "flow1_tracking_score.csv", "Download flow1_tracking_score.csv", _safe_key(case_id, "download", "flow1_score_file"))
    with score_cols[1]:
        _download_file(run_dir / "flow2_global_id_score.csv", "Download flow2_global_id_score.csv", _safe_key(case_id, "download", "flow2_score_file"))
    with score_cols[2]:
        _download_file(run_dir / "final_pipeline_score.csv", "Download final_pipeline_score.csv", _safe_key(case_id, "download", "final_score_file"))


def cleanup_panel(case: dict, state: dict) -> None:
    case_id = case["case_id"]
    with st.expander("Cleanup output", expanded=False):
        st.caption("Gunakan tombol ini saat localhost mulai berat atau output lama sudah tidak diperlukan.")
        confirm_current = st.checkbox("Konfirmasi hapus output run aktif", key=_safe_key(case_id, "confirm_clear_current"))
        if st.button("Bersihkan output run aktif", disabled=not confirm_current, key=_safe_key(case_id, "clear_current_run")):
            ok = clear_current_run_output(case_id)
            st.success("Output run aktif dibersihkan." if ok else "Tidak ada run aktif.")
            st.rerun()

        confirm_case = st.checkbox("Konfirmasi hapus semua output case ini", key=_safe_key(case_id, "confirm_clear_case"))
        if st.button("Bersihkan semua output case ini", disabled=not confirm_case, key=_safe_key(case_id, "clear_case_outputs")):
            clear_selected_case_outputs(case_id)
            st.success("Semua output case terpilih dibersihkan.")
            st.rerun()


def reid_panel_tabs(case: dict, state: dict, use_cuda: bool, osnet_weight: str | None) -> None:
    case_id = case["case_id"]
    st.markdown("### Tahap 2 - Re-ID dan association")
    is_multicamera = len(case.get("cameras", [])) > 1
    if is_multicamera:
        st.info(
            "Tracking config dapat berbeda per kamera, tetapi Re-ID config tetap case-level "
            "karena Global ID dibentuk dari seluruh valid track."
        )
        ready = _camera_valid_ready(case, state.get("run_dir"))
        valid_all_ready = bool(state.get("run_dir")) and (Path(state["run_dir"]) / "valid_tracks_all.csv").exists()
        checklist = pd.DataFrame([
            {"item": f"{camera} valid track tersedia", "ready": "ya" if ok else "tidak"}
            for camera, ok in ready.items()
        ] + [{"item": "valid_tracks_all.csv siap", "ready": "ya" if valid_all_ready else "tidak"}])
        st.markdown("#### Kesiapan Re-ID")
        st.dataframe(checklist, use_container_width=True, hide_index=True)
        state["tracking_done"] = bool(ready) and all(ready.values()) and valid_all_ready

    if not state.get("tracking_done"):
        st.info("Jalankan tracking sampai valid track tersedia sebelum menjalankan Re-ID.")
        return

    cfg = reid_config_panel(case, state)
    state["config"] = normalize_config(cfg)
    _set_runtime_config_from_state(state)
    run = st.button("Jalankan Re-ID", type="primary", key=_safe_key(case_id, "run_reid_tabs"), disabled=not state.get("tracking_done"))

    if run:
        reid_progress = st.progress(0)
        reid_status = st.empty()

        def on_reid(done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            reid_progress.progress(min(100, pct))
            reid_status.write(f"Ekstraksi embedding: {done}/{total} crop ({pct}%)")

        try:
            start = time.perf_counter()
            result = run_reid_stage(
                case=case,
                config=state["runtime_config"],
                run_dir=state["run_dir"],
                osnet_weight=osnet_weight,
                use_cuda=use_cuda,
                reid_progress_callback=on_reid,
                render=False,
            )
            result["runtime_sec"] = time.perf_counter() - start
            state["reid_done"] = True
            state["render_done"] = False
            state["reid_result"] = result
            reid_progress.progress(100)
            reid_status.success("Re-ID selesai.")
            for message in result.get("stage_log", []):
                st.info(message)
        except Exception as e:
            st.error(f"Re-ID gagal: {e}")
            st.exception(e)

    if not (state.get("reid_done") and state.get("reid_result")):
        return

    result = state["reid_result"]
    if result.get("stage_log"):
        st.caption("Stage log: " + " | ".join(map(str, result.get("stage_log", []))))
    if result.get("reid_config_used"):
        with st.expander("Re-ID config used", expanded=False):
            st.json(result["reid_config_used"])
    metrics = _reid_summary_metrics(result, result.get("runtime_sec"))
    reid_score = _reid_score_df(result, metrics, state.get("tracking_result"))
    tracking_score = state.get("tracking_score_df")
    if tracking_score is None and state.get("tracking_result"):
        tracking_metrics = _tracking_metrics_df(state["tracking_result"])
        tracking_score = _tracking_score_df(state["tracking_result"], tracking_metrics)
        state["tracking_score_df"] = tracking_score
    state["reid_score_df"] = reid_score
    state["final_score_df"] = _final_score_df(tracking_score, reid_score)

    pair_df = result.get("pair_df", pd.DataFrame())
    global_meta = result.get("global_meta_df", pd.DataFrame())
    merged_pairs = _merged_pairs_table(pair_df, global_meta)
    intra_count = int((merged_pairs.get("merge_type") == "intra-camera").sum()) if len(merged_pairs) and "merge_type" in merged_pairs else 0
    cross_count = int((merged_pairs.get("merge_type") == "cross-camera").sum()) if len(merged_pairs) and "merge_type" in merged_pairs else 0
    row = metrics.iloc[0].to_dict()

    tabs = st.tabs([
        "Ringkasan Re-ID",
        "Pasangan tergabung",
        "Bukti merge track",
        "Similarity kandidat",
        "Evaluasi Re-ID",
        "Unduhan",
    ])

    with tabs[0]:
        st.caption("Tahap 2 menunjukkan hasil association antar-track berdasarkan embedding OSNet dan cosine similarity.")
        cols = st.columns(6)
        cols[0].metric("Local track", int(row["num_local_tracks"]))
        cols[1].metric("Global ID sementara", int(row["num_global_ids"]))
        cols[2].metric("Merged pairs", int(len(merged_pairs)))
        cols[3].metric("Intra-camera", intra_count)
        cols[4].metric("Cross-camera", cross_count)
        cols[5].metric("Fragment reduction", f"{row['fragment_reduction_rate']:.2%}")
        if bool(reid_score.iloc[0].get("score_available", False)):
            st.metric("Re-ID score", f"{float(reid_score.iloc[0]['reid_score']):.3f}")
        else:
            st.info(str(reid_score.iloc[0].get("reason", "Re-ID score tidak tersedia.")))
        _show_table(
            reid_score,
            "Re-ID score",
            "flow2_global_id_score.csv",
            _safe_key(case_id, "download", "flow2_global_id_score_tabs"),
            run_dir=result["run_dir"],
        )

    with tabs[1]:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total pasangan", int(len(merged_pairs)))
        metric_cols[1].metric("Intra-camera", intra_count)
        metric_cols[2].metric("Cross-camera", cross_count)
        _show_table(
            merged_pairs,
            "Pasangan track tergabung",
            f"{case_id}_merged_pairs.csv",
            _safe_key(case_id, "download", "merged_pairs_tabs"),
            ["global_id", "track_a", "track_b", "camera_a", "camera_b", "merge_type", "cosine_similarity", "merge_reason"],
            run_dir=result["run_dir"],
        )

    with tabs[2]:
        _show_merge_gallery(result, case_id)

    with tabs[3]:
        top_n = st.slider("Jumlah kandidat ditampilkan", 5, 100, 20, 5, key=_safe_key(case_id, "top_similarity_n_tabs"))
        _show_table(
            _top_similarity_table(pair_df, top_n),
            "Kandidat similarity tertinggi",
            f"{case_id}_top_similarity_candidates.csv",
            _safe_key(case_id, "download", "top_similarity_tabs"),
            ["track_a", "track_b", "camera_a", "camera_b", "same_camera", "temporal_gap", "temporal_overlap", "cosine_similarity", "merge_status", "reject_reason"],
            run_dir=result["run_dir"],
        )

    with tabs[4]:
        pairwise_eval = result.get("reid_pairwise_eval_df", pd.DataFrame())
        pairwise_detail = result.get("reid_pairwise_detail_df", pd.DataFrame())
        if pairwise_eval is not None and len(pairwise_eval):
            eval_row = pairwise_eval.iloc[0].to_dict()
            assoc = bool(eval_row.get("association_eval_available", False))
            if assoc:
                metric_cols = st.columns(5)
                metric_cols[0].metric("Pairwise precision", "N/A" if pd.isna(eval_row.get("pairwise_precision")) else f"{float(eval_row.get('pairwise_precision')):.3f}")
                metric_cols[1].metric("Pairwise recall", "N/A" if pd.isna(eval_row.get("pairwise_recall")) else f"{float(eval_row.get('pairwise_recall')):.3f}")
                metric_cols[2].metric("Pairwise F1", "N/A" if pd.isna(eval_row.get("pairwise_f1")) else f"{float(eval_row.get('pairwise_f1')):.3f}")
                metric_cols[3].metric("False merge", "N/A" if pd.isna(eval_row.get("false_merge_rate")) else f"{float(eval_row.get('false_merge_rate')):.3f}")
                metric_cols[4].metric("False split", "N/A" if pd.isna(eval_row.get("false_split_rate")) else f"{float(eval_row.get('false_split_rate')):.3f}")
                st.dataframe(_pairwise_confusion_matrix_df(pairwise_eval), use_container_width=True, hide_index=True)
            else:
                st.info("Ground Truth tidak tersedia. Evaluasi Re-ID berbasis GT tidak aktif.")
            _show_table(
                pairwise_eval,
                "Evaluasi pairwise Re-ID",
                "reid_pairwise_evaluation.csv",
                _safe_key(case_id, "download", "reid_pairwise_evaluation_tabs"),
                run_dir=result["run_dir"],
            )
            with st.expander("Detail evaluasi pairwise", expanded=False):
                _show_table(
                    pairwise_detail,
                    "Detail pairwise",
                    "reid_pairwise_detail.csv",
                    _safe_key(case_id, "download", "reid_pairwise_detail_tabs"),
                    [
                        "track_a",
                        "track_b",
                        "camera_a",
                        "camera_b",
                        "global_id_a",
                        "global_id_b",
                        "dominant_gt_id_a",
                        "dominant_gt_id_b",
                        "track_purity_a",
                        "track_purity_b",
                        "same_gt",
                        "same_global_id",
                        "pair_eval_type",
                        "cosine_similarity",
                        "temporal_gap",
                        "temporal_overlap",
                        "merge_status",
                        "merge_reason",
                        "reject_reason",
                    ],
                    run_dir=result["run_dir"],
                )
        else:
            st.info("Ground Truth tidak tersedia. Evaluasi Re-ID berbasis GT tidak aktif.")

    with tabs[5]:
        for filename in [
            "pair_similarity.csv",
            "merged_pairs.csv",
            "global_track_meta.csv",
            "global_id_evaluation.csv",
            "global_id_summary_metrics.csv",
            "reid_pairwise_evaluation.csv",
            "reid_pairwise_detail.csv",
            "reid_diagnostics.csv",
            "reid_tuning_hints.csv",
            "flow2_global_id_score.csv",
        ]:
            path = Path(result["run_dir"]) / filename
            if path.exists():
                _download_file(path, f"Unduh {filename}", _safe_key(case_id, "download_file_tabs", filename))
        merge_index = Path(result["run_dir"]) / "merge_gallery" / "merge_gallery_index.csv"
        if merge_index.exists():
            _download_file(merge_index, "Unduh merge_gallery_index.csv", _safe_key(case_id, "download_file_tabs", "merge_gallery_index"))


def render_output_panel_tabs(case: dict, state: dict) -> None:
    case_id = case["case_id"]
    st.markdown("### Tahap 3 - Render dan hasil final")
    if not state.get("reid_done"):
        st.info("Render video setelah Global ID selesai dibentuk.")
        return

    result = state["reid_result"]
    run_dir = Path(state["run_dir"])
    combined_layout = "fullscreen_grid"
    st.caption("Video gabungan memakai layout fullscreen 1920x1080.")
    if st.button("Render video", type="primary", key=_safe_key(case_id, "render_video_tabs")):
        progress = st.progress(0)
        status = st.empty()

        def on_render(camera: str, done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            progress.progress(min(100, pct))
            status.write(f"Render {camera}: {done}/{total} frame ({pct}%)")

        try:
            outputs = render_reid_outputs(
                case,
                run_dir,
                result.get("render_df"),
                on_render,
                combined_layout=combined_layout,
            )
            result.update(outputs)
            state["reid_result"] = result
            state["render_done"] = True
            progress.progress(100)
            status.success("Render selesai.")
        except Exception as e:
            st.error(f"Render gagal: {e}")
            st.exception(e)

    tracking_score = state.get("tracking_score_df", pd.DataFrame())
    reid_score = state.get("reid_score_df", pd.DataFrame())
    final_score = state.get("final_score_df", pd.DataFrame())
    global_summary = _global_summary_table(result)
    meta = result.get("global_meta_df", pd.DataFrame())
    merged_pairs = _merged_pairs_table(result.get("pair_df", pd.DataFrame()), meta)
    tracking_result = state.get("tracking_result") or {}
    local_count = int(meta["track_key"].nunique()) if meta is not None and len(meta) and "track_key" in meta else 0
    valid_count = _valid_track_count(tracking_result) if tracking_result else local_count

    tabs = st.tabs(["Ringkasan final", "Video hasil", "Galeri Global ID", "Detail Global ID", "Unduhan"])

    with tabs[0]:
        st.caption("Hasil final menggabungkan tracking lokal dan Re-ID menjadi Global ID yang ditampilkan pada video dan gallery.")
        cols = st.columns(4)
        if final_score is not None and len(final_score) and bool(final_score.iloc[0].get("score_available", False)):
            cols[0].metric("Final pipeline score", f"{float(final_score.iloc[0]['final_pipeline_score']):.3f}")
        else:
            cols[0].metric("Final pipeline score", "N/A")
        if tracking_score is not None and len(tracking_score) and bool(tracking_score.iloc[0].get("score_available", False)):
            cols[1].metric("Tracking score", f"{float(tracking_score.iloc[0]['tracking_score']):.3f}")
        else:
            cols[1].metric("Tracking score", "N/A")
        if reid_score is not None and len(reid_score) and bool(reid_score.iloc[0].get("score_available", False)):
            cols[2].metric("Re-ID score", f"{float(reid_score.iloc[0]['reid_score']):.3f}")
        else:
            cols[2].metric("Re-ID score", "N/A")
        cols[3].metric("Global ID final", int(global_summary["global_id"].nunique()) if len(global_summary) and "global_id" in global_summary else 0)
        cols = st.columns(4)
        cols[0].metric("Local track", local_count)
        cols[1].metric("Valid track", valid_count)
        cols[2].metric("Merged pairs", int(len(merged_pairs)))
        cols[3].metric("Jumlah kamera", len(case.get("cameras", [])))

    with tabs[1]:
        if result.get("combined_path") and Path(result["combined_path"]).exists():
            st.subheader("Video gabungan")
            st.video(str(result["combined_path"]))
            _download_file(result["combined_path"], "Unduh video gabungan", _safe_key(case_id, "download", "combined_video_tabs"))
        else:
            st.info("Video gabungan belum dibuat. Klik Render video untuk membuat output final.")
        with st.expander("Video per kamera", expanded=False):
            for path in result.get("rendered_paths", []) or []:
                p = Path(path)
                if p.exists():
                    st.markdown(f"#### {p.stem}")
                    st.video(str(p))
                    _download_file(p, f"Unduh {p.name}", _safe_key(case_id, "download", "camera_video_tabs", p.name))

    with tabs[2]:
        _show_global_gallery_filtered(result, case_id)

    with tabs[3]:
        _show_table(
            global_summary,
            "Ringkasan Global ID",
            f"{case_id}_ringkasan_global_id.csv",
            _safe_key(case_id, "download", "global_id_summary_tabs"),
            ["global_id", "num_tracks", "members", "cameras", "first_frame", "last_frame", "dominant_gt_id", "global_id_purity", "is_mixed_gid"],
            run_dir=run_dir,
        )

    with tabs[4]:
        downloads = [
            "global_track_meta.csv",
            "merged_pairs.csv",
            "pair_similarity.csv",
            "run_config.json",
            "final_pipeline_score.csv",
            "flow1_tracking_score.csv",
            "flow2_global_id_score.csv",
        ]
        for filename in downloads:
            path = run_dir / filename
            if path.exists():
                _download_file(path, f"Unduh {filename}", _safe_key(case_id, "download_final", filename))
        if result.get("combined_path"):
            _download_file(result["combined_path"], "Unduh combined_multicamera_rendered.mp4", _safe_key(case_id, "download_final", "combined_video"))
        for path in result.get("rendered_paths", []) or []:
            p = Path(path)
            if p.exists():
                _download_file(p, f"Unduh {p.name}", _safe_key(case_id, "download_final", p.name))


def case_tab(case: dict, use_cuda: bool, yolo_weight: str | None, osnet_weight: str | None) -> None:
    case = with_gt_meta(case)
    state = get_case_state(case["case_id"])
    _case_overview(case)

    with st.expander("Status video dan annotation", expanded=False):
        asset_status(case)
    with st.expander("Preview video", expanded=False):
        preview_videos(case)
    cleanup_panel(case, state)

    st.divider()
    tracking_panel(case, state, use_cuda, yolo_weight)
    st.divider()
    reid_panel_tabs(case, state, use_cuda, osnet_weight)
    st.divider()
    render_output_panel_tabs(case, state)


def main() -> None:
    st.set_page_config(page_title="Person Re-ID Multi-Kamera", layout="wide")
    apply_styles()
    init_state()
    hero()

    status = cuda_status()
    st.sidebar.subheader("Status perangkat")
    if status["cuda_available"]:
        st.sidebar.success(f"CUDA aktif: {status['device_name']}")
    else:
        st.sidebar.warning("CUDA tidak tersedia. Sistem berjalan menggunakan CPU.")

    with st.sidebar.expander("Detail CUDA"):
        st.json(status)

    st.sidebar.subheader("Model")
    st.sidebar.write("Model deteksi: YOLO11n")
    st.sidebar.write("Model Re-ID: OSNet")
    yolo_weight = str(YOLO_LOCAL_PATH)
    osnet_weight = str(OSNET_DEFAULT_PATH)
    model_ok = Path(yolo_weight).exists() and Path(osnet_weight).exists()
    st.sidebar.success("Status model: tersedia" if model_ok else "Status model: belum lengkap")
    with st.sidebar.expander("Detail model"):
        st.write(f"YOLO: {yolo_weight}")
        st.write(f"OSNet: {osnet_weight}")
    use_cuda = bool(status["cuda_available"])

    if st.sidebar.button("Reset semua case", key="reset_all_cases"):
        reset_all(delete_output=False)
        st.rerun()

    st.sidebar.subheader("Cleanup")
    keep_latest = st.sidebar.number_input("Simpan run terbaru per case", 1, 20, 3, 1, key="keep_latest_runs")
    confirm_old = st.sidebar.checkbox("Konfirmasi bersihkan run lama", key="confirm_clear_old_runs")
    if st.sidebar.button("Bersihkan run lama", disabled=not confirm_old, key="clear_all_old_runs"):
        removed = clear_all_old_runs(int(keep_latest))
        st.sidebar.success(f"Run lama dibersihkan: {removed} folder.")
        st.rerun()

    confirm_cache = st.sidebar.checkbox("Konfirmasi bersihkan cache Streamlit", key="confirm_clear_cache")
    if st.sidebar.button("Bersihkan cache Streamlit", disabled=not confirm_cache, key="clear_streamlit_cache"):
        clear_streamlit_cache_and_state()
        st.sidebar.success("Streamlit cache dan session run state dibersihkan.")
        st.rerun()

    tabs = st.tabs([_case_label(case) for case in DEMO_CASES])
    for tab, case in zip(tabs, DEMO_CASES):
        with tab:
            case_tab(case, use_cuda, yolo_weight, osnet_weight)


if __name__ == "__main__":
    main()
