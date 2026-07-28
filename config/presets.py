from __future__ import annotations

from copy import deepcopy

from config.cases import get_case_by_id


TRACKING_DEFAULTS = {
    "yolo_conf": 0.05,
    "yolo_iou": 0.50,
    "imgsz": 960,
    "track_high_thresh": 0.12,
    "track_low_thresh": 0.03,
    "new_track_thresh": 0.12,
    "match_thresh": 0.80,
    "track_buffer": 45,
}

FILTER_DEFAULTS = {
    "min_frames": 10,
    "min_crops": 10,
    "min_avg_conf": 0.20,
    "min_avg_area": 800.0,
    "max_samples_per_track": 32,
}

REID_DEFAULTS = {
    "enable_cross_camera": True,
    "enable_strict_intra": True,
    "cross_threshold": 0.80,
    "intra_threshold": 0.82,
    "intra_max_gap": 30,
    "intra_max_overlap": 0,
    "use_mnn": True,
}

TRACKING_KEYS = tuple(TRACKING_DEFAULTS)
FILTER_KEYS = tuple(FILTER_DEFAULTS)
REID_KEYS = tuple(REID_DEFAULTS)
ALLOWED_IMGSZ = (640, 960, 1280)


INITIAL_ANALYSIS_CONFIG = {
    "tracking": deepcopy(TRACKING_DEFAULTS),
    "filter": deepcopy(FILTER_DEFAULTS),
    "reid": deepcopy(REID_DEFAULTS),
}


CASE_RECOMMENDATIONS = {
    "case_1_normal_success": {
        "camera_configs": {
            "camera_2": {
                "tracking": {"yolo_conf": 0.05, "yolo_iou": 0.50, "imgsz": 960, "track_high_thresh": 0.12, "track_low_thresh": 0.03, "new_track_thresh": 0.12, "match_thresh": 0.80, "track_buffer": 45},
                "filter": {"min_frames": 10, "min_crops": 10, "min_avg_conf": 0.20, "min_avg_area": 800.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": False, "enable_strict_intra": False, "cross_threshold": 0.00, "intra_threshold": 0.00, "intra_max_gap": 0, "intra_max_overlap": 0, "use_mnn": False},
    },
    "case_2_crowded_fragmentation": {
        "camera_configs": {
            "camera_3": {
                "tracking": {"yolo_conf": 0.20, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.20, "track_low_thresh": 0.05, "new_track_thresh": 0.20, "match_thresh": 0.80, "track_buffer": 45},
                "filter": {"min_frames": 80, "min_crops": 80, "min_avg_conf": 0.40, "min_avg_area": 3000.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": False, "enable_strict_intra": True, "cross_threshold": 0.00, "intra_threshold": 0.78, "intra_max_gap": 30, "intra_max_overlap": 0, "use_mnn": False},
    },
    "case_3_failure_limitation": {
        "camera_configs": {
            "camera_1": {
                "tracking": {"yolo_conf": 0.03, "yolo_iou": 0.60, "imgsz": 1280, "track_high_thresh": 0.075, "track_low_thresh": 0.03, "new_track_thresh": 0.075, "match_thresh": 0.84, "track_buffer": 30},
                "filter": {"min_frames": 15, "min_crops": 15, "min_avg_conf": 0.25, "min_avg_area": 500.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": False, "enable_strict_intra": True, "cross_threshold": 0.00, "intra_threshold": 0.77, "intra_max_gap": 300, "intra_max_overlap": 60, "use_mnn": False},
    },
    "case_4_multicamera_success": {
        "camera_configs": {
            "camera_1": {
                "tracking": {"yolo_conf": 0.05, "yolo_iou": 0.50, "imgsz": 960, "track_high_thresh": 0.12, "track_low_thresh": 0.03, "new_track_thresh": 0.12, "match_thresh": 0.80, "track_buffer": 45},
                "filter": {"min_frames": 20, "min_crops": 20, "min_avg_conf": 0.20, "min_avg_area": 800.0, "max_samples_per_track": 32},
            },
            "camera_3": {
                "tracking": {"yolo_conf": 0.05, "yolo_iou": 0.50, "imgsz": 960, "track_high_thresh": 0.12, "track_low_thresh": 0.03, "new_track_thresh": 0.12, "match_thresh": 0.80, "track_buffer": 45},
                "filter": {"min_frames": 10, "min_crops": 10, "min_avg_conf": 0.20, "min_avg_area": 800.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": True, "enable_strict_intra": True, "cross_threshold": 0.77, "intra_threshold": 0.78, "intra_max_gap": 584, "intra_max_overlap": 10, "use_mnn": True},
    },
    "case_5_multicamera_3_video_stress": {
        "camera_configs": {
            "camera_1": {
                "tracking": {"yolo_conf": 0.25, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.25, "track_low_thresh": 0.10, "new_track_thresh": 0.25, "match_thresh": 0.78, "track_buffer": 30},
                "filter": {"min_frames": 20, "min_crops": 20, "min_avg_conf": 0.35, "min_avg_area": 4000.0, "max_samples_per_track": 32},
            },
            "camera_2": {
                "tracking": {"yolo_conf": 0.20, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.20, "track_low_thresh": 0.10, "new_track_thresh": 0.20, "match_thresh": 0.80, "track_buffer": 30},
                "filter": {"min_frames": 100, "min_crops": 100, "min_avg_conf": 0.50, "min_avg_area": 8000.0, "max_samples_per_track": 32},
            },
            "camera_3": {
                "tracking": {"yolo_conf": 0.20, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.20, "track_low_thresh": 0.05, "new_track_thresh": 0.20, "match_thresh": 0.80, "track_buffer": 45},
                "filter": {"min_frames": 80, "min_crops": 80, "min_avg_conf": 0.40, "min_avg_area": 3000.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": True, "enable_strict_intra": True, "cross_threshold": 0.80, "intra_threshold": 0.78, "intra_max_gap": 30, "intra_max_overlap": 0, "use_mnn": True},
    },
    "case_6_multicamera_temporal_handoff": {
        "camera_configs": {
            "camera_6": {
                "tracking": {"yolo_conf": 0.20, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.20, "track_low_thresh": 0.07, "new_track_thresh": 0.20, "match_thresh": 0.80, "track_buffer": 30},
                "filter": {"min_frames": 60, "min_crops": 60, "min_avg_conf": 0.55, "min_avg_area": 15000.0, "max_samples_per_track": 32},
            },
            "camera_5": {
                "tracking": {"yolo_conf": 0.20, "yolo_iou": 0.50, "imgsz": 640, "track_high_thresh": 0.20, "track_low_thresh": 0.07, "new_track_thresh": 0.20, "match_thresh": 0.80, "track_buffer": 30},
                "filter": {"min_frames": 45, "min_crops": 45, "min_avg_conf": 0.55, "min_avg_area": 15000.0, "max_samples_per_track": 32},
            },
        },
        "reid": {"enable_cross_camera": True, "enable_strict_intra": False, "cross_threshold": 0.80, "intra_threshold": 0.00, "intra_max_gap": 0, "intra_max_overlap": 0, "use_mnn": True},
    },
}


def _known_values(values: dict | None, keys: tuple[str, ...]) -> dict:
    values = values or {}
    return {key: deepcopy(values[key]) for key in keys if key in values}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"boolean value is invalid: {value}")
    return bool(value)


def _cast_tracking(values: dict) -> dict:
    casts = {"yolo_conf": float, "yolo_iou": float, "imgsz": int, "track_high_thresh": float,
             "track_low_thresh": float, "new_track_thresh": float, "match_thresh": float, "track_buffer": int}
    return {key: casts[key](values[key]) for key in TRACKING_KEYS}


def _cast_filter(values: dict) -> dict:
    casts = {"min_frames": int, "min_crops": int, "min_avg_conf": float, "min_avg_area": float,
             "max_samples_per_track": int}
    return {key: casts[key](values[key]) for key in FILTER_KEYS}


def _cast_reid(values: dict) -> dict:
    casts = {"enable_cross_camera": _as_bool, "enable_strict_intra": _as_bool, "cross_threshold": float,
             "intra_threshold": float, "intra_max_gap": int, "intra_max_overlap": int, "use_mnn": _as_bool}
    return {key: casts[key](values[key]) for key in REID_KEYS}


def _validate_config(config: dict) -> None:
    tracking = config["tracking"]
    filtering = config["filter"]
    reid = config["reid"]
    for key in ("yolo_conf", "yolo_iou", "track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        if not 0.0 < tracking[key] <= 1.0:
            raise ValueError(f"tracking.{key} must be greater than 0 and at most 1")
    if tracking["imgsz"] not in ALLOWED_IMGSZ:
        raise ValueError(f"tracking.imgsz must be one of {list(ALLOWED_IMGSZ)}")
    if tracking["track_low_thresh"] > tracking["track_high_thresh"]:
        raise ValueError("tracking.track_low_thresh cannot exceed track_high_thresh")
    if tracking["track_buffer"] < 0:
        raise ValueError("tracking.track_buffer cannot be negative")
    if filtering["min_frames"] < 1:
        raise ValueError("filter.min_frames must be at least 1")
    if filtering["min_crops"] < 1:
        raise ValueError("filter.min_crops must be at least 1")
    if not 0.0 <= filtering["min_avg_conf"] <= 1.0:
        raise ValueError("filter.min_avg_conf must be between 0 and 1")
    if filtering["min_avg_area"] <= 0:
        raise ValueError("filter.min_avg_area must be greater than 0")
    if filtering["max_samples_per_track"] < 1:
        raise ValueError("filter.max_samples_per_track must be at least 1")
    for key in ("cross_threshold", "intra_threshold"):
        if not 0.0 <= reid[key] <= 1.0:
            raise ValueError(f"reid.{key} must be between 0 and 1")
    for key in ("intra_max_gap", "intra_max_overlap"):
        if reid[key] < 0:
            raise ValueError(f"reid.{key} cannot be negative")


def normalize_config(config: dict | None) -> dict:
    """Return an independent configuration using the canonical runtime schema."""
    source = deepcopy(config or {})
    normalized = {
        "config_mode": str(source.get("config_mode", "initial_analysis")),
        "tracking": deepcopy(TRACKING_DEFAULTS),
        "filter": deepcopy(FILTER_DEFAULTS),
        "reid": deepcopy(REID_DEFAULTS),
        "camera_configs": {},
    }
    if normalized["config_mode"] not in {"initial_analysis", "case_recommendation"}:
        raise ValueError("config_mode must be 'initial_analysis' or 'case_recommendation'")
    normalized["tracking"].update(_known_values(source.get("tracking"), TRACKING_KEYS))
    normalized["filter"].update(_known_values(source.get("filter"), FILTER_KEYS))
    normalized["reid"].update(_known_values(source.get("reid"), REID_KEYS))
    normalized["tracking"] = _cast_tracking(normalized["tracking"])
    normalized["filter"] = _cast_filter(normalized["filter"])
    normalized["reid"] = _cast_reid(normalized["reid"])

    for camera, camera_override in (source.get("camera_configs") or {}).items():
        camera_override = camera_override or {}
        camera_tracking = deepcopy(normalized["tracking"])
        camera_filter = deepcopy(normalized["filter"])
        camera_tracking.update(_known_values(camera_override.get("tracking"), TRACKING_KEYS))
        camera_filter.update(_known_values(camera_override.get("filter"), FILTER_KEYS))
        normalized["camera_configs"][str(camera)] = {
            "tracking": _cast_tracking(camera_tracking),
            "filter": _cast_filter(camera_filter),
        }

    _validate_config(normalized)
    for camera_config in normalized["camera_configs"].values():
        _validate_config({**normalized, **camera_config})
    return normalized


def build_initial_analysis_config(case_id: str) -> dict:
    case = get_case_by_id(case_id)
    config = {
        "config_mode": "initial_analysis",
        "tracking": deepcopy(TRACKING_DEFAULTS),
        "filter": deepcopy(FILTER_DEFAULTS),
        "reid": deepcopy(REID_DEFAULTS),
        "camera_configs": {
            camera: {"tracking": deepcopy(TRACKING_DEFAULTS), "filter": deepcopy(FILTER_DEFAULTS)}
            for camera in case["cameras"]
        },
    }
    return normalize_config(config)


def build_recommended_config(case_id: str) -> dict:
    case = get_case_by_id(case_id)
    recommendation = deepcopy(CASE_RECOMMENDATIONS[case_id])
    camera_configs = recommendation["camera_configs"]
    first_camera = case["cameras"][0]
    config = {
        "config_mode": "case_recommendation",
        "tracking": deepcopy(camera_configs[first_camera]["tracking"]),
        "filter": deepcopy(camera_configs[first_camera]["filter"]),
        "reid": deepcopy(recommendation["reid"]),
        "camera_configs": camera_configs,
    }
    return normalize_config(config)


def get_camera_stage_config(config: dict | None, camera_name: str) -> tuple[dict, dict]:
    """Return independent tracking and filter settings for one camera."""
    normalized = normalize_config(config)
    camera_config = normalized["camera_configs"].get(camera_name)
    if camera_config is None:
        return deepcopy(normalized["tracking"]), deepcopy(normalized["filter"])
    return deepcopy(camera_config["tracking"]), deepcopy(camera_config["filter"])
