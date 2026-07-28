from config.cases import DEMO_CASES
from config.presets import build_initial_analysis_config, build_recommended_config, normalize_config


TRACKING_KEYS = (
    "yolo_conf", "yolo_iou", "imgsz", "track_high_thresh", "track_low_thresh",
    "new_track_thresh", "match_thresh", "track_buffer",
)
FILTER_KEYS = ("min_frames", "min_crops", "min_avg_conf", "min_avg_area", "max_samples_per_track")
REID_KEYS = (
    "enable_cross_camera", "enable_strict_intra", "cross_threshold", "intra_threshold",
    "intra_max_gap", "intra_max_overlap", "use_mnn",
)


INITIAL_SNAPSHOT = {
    "tracking": (0.05, 0.50, 960, 0.12, 0.03, 0.12, 0.80, 45),
    "filter": (10, 10, 0.20, 800.0, 32),
    "reid": (True, True, 0.80, 0.82, 30, 0, True),
}


RECOMMENDATION_SNAPSHOT = {
    "case_1_normal_success": {
        "camera_2": ((0.05, 0.50, 960, 0.12, 0.03, 0.12, 0.80, 45), (10, 10, 0.20, 800.0, 32)),
        "reid": (False, False, 0.00, 0.00, 0, 0, False),
    },
    "case_2_crowded_fragmentation": {
        "camera_3": ((0.20, 0.50, 640, 0.20, 0.05, 0.20, 0.80, 45), (80, 80, 0.40, 3000.0, 32)),
        "reid": (False, True, 0.00, 0.78, 30, 0, False),
    },
    "case_3_failure_limitation": {
        "camera_1": ((0.03, 0.60, 1280, 0.075, 0.03, 0.075, 0.84, 30), (15, 15, 0.25, 500.0, 32)),
        "reid": (False, True, 0.00, 0.77, 300, 60, False),
    },
    "case_4_multicamera_success": {
        "camera_1": ((0.05, 0.50, 960, 0.12, 0.03, 0.12, 0.80, 45), (20, 20, 0.20, 800.0, 32)),
        "camera_3": ((0.05, 0.50, 960, 0.12, 0.03, 0.12, 0.80, 45), (10, 10, 0.20, 800.0, 32)),
        "reid": (True, True, 0.77, 0.78, 584, 10, True),
    },
    "case_5_multicamera_3_video_stress": {
        "camera_1": ((0.25, 0.50, 640, 0.25, 0.10, 0.25, 0.78, 30), (20, 20, 0.35, 4000.0, 32)),
        "camera_2": ((0.20, 0.50, 640, 0.20, 0.10, 0.20, 0.80, 30), (100, 100, 0.50, 8000.0, 32)),
        "camera_3": ((0.20, 0.50, 640, 0.20, 0.05, 0.20, 0.80, 45), (80, 80, 0.40, 3000.0, 32)),
        "reid": (True, True, 0.80, 0.78, 30, 0, True),
    },
    "case_6_multicamera_temporal_handoff": {
        "camera_6": ((0.20, 0.50, 640, 0.20, 0.07, 0.20, 0.80, 30), (60, 60, 0.55, 15000.0, 32)),
        "camera_5": ((0.20, 0.50, 640, 0.20, 0.07, 0.20, 0.80, 30), (45, 45, 0.55, 15000.0, 32)),
        "reid": (True, False, 0.80, 0.00, 0, 0, True),
    },
}


def _values(section, keys):
    return tuple(section[key] for key in keys)


def test_configuration_snapshot_preserves_active_initial_and_recommendation_values():
    for case in DEMO_CASES:
        initial = build_initial_analysis_config(case["case_id"])
        assert _values(initial["tracking"], TRACKING_KEYS) == INITIAL_SNAPSHOT["tracking"]
        assert _values(initial["filter"], FILTER_KEYS) == INITIAL_SNAPSHOT["filter"]
        assert _values(initial["reid"], REID_KEYS) == INITIAL_SNAPSHOT["reid"]
        assert list(initial["camera_configs"]) == case["cameras"]

        recommended = build_recommended_config(case["case_id"])
        expected = RECOMMENDATION_SNAPSHOT[case["case_id"]]
        assert _values(recommended["reid"], REID_KEYS) == expected["reid"]
        assert list(recommended["camera_configs"]) == case["cameras"]
        for camera in case["cameras"]:
            tracking, filtering = expected[camera]
            assert _values(recommended["camera_configs"][camera]["tracking"], TRACKING_KEYS) == tracking
            assert _values(recommended["camera_configs"][camera]["filter"], FILTER_KEYS) == filtering


def test_custom_config_snapshot_is_not_overwritten_by_defaults():
    custom = normalize_config({
        "config_mode": "initial_analysis",
        "tracking": {"yolo_conf": 0.22, "yolo_iou": 0.61, "imgsz": 1280, "track_buffer": 91},
        "filter": {"min_frames": 17, "max_samples_per_track": 19},
        "reid": {"cross_threshold": 0.73, "intra_threshold": 0.71},
        "camera_configs": {"camera_2": {"tracking": {"yolo_conf": 0.33}, "filter": {"min_frames": 23}}},
    })
    assert custom["tracking"]["yolo_conf"] == 0.22
    assert custom["tracking"]["yolo_iou"] == 0.61
    assert custom["tracking"]["imgsz"] == 1280
    assert custom["tracking"]["track_buffer"] == 91
    assert custom["filter"]["min_frames"] == 17
    assert custom["filter"]["max_samples_per_track"] == 19
    assert custom["reid"]["cross_threshold"] == 0.73
    assert custom["reid"]["intra_threshold"] == 0.71
    assert custom["camera_configs"]["camera_2"]["tracking"]["yolo_conf"] == 0.33
    assert custom["camera_configs"]["camera_2"]["filter"]["min_frames"] == 23


def test_canonical_config_has_no_legacy_fields_or_shared_camera_dictionaries():
    legacy_fields = {
        "tracking_preset", "filter_preset", "reid_preset", "visual_preset", "profile",
        "recommended_config", "recommended_note", "expected_result", "tracking_config_mode",
        "crop_selection_strategy", "scene_type", "custom",
    }
    config = build_initial_analysis_config("case_4_multicamera_success")
    assert not legacy_fields.intersection(config)
    assert not legacy_fields.intersection(config["filter"])
    for camera_config in config["camera_configs"].values():
        assert not legacy_fields.intersection(camera_config)
        assert not legacy_fields.intersection(camera_config["filter"])
    config["camera_configs"]["camera_1"]["tracking"]["yolo_conf"] = 0.31
    assert config["camera_configs"]["camera_3"]["tracking"]["yolo_conf"] == 0.05
    other_case = build_initial_analysis_config("case_5_multicamera_3_video_stress")
    other_case["camera_configs"]["camera_1"]["filter"]["min_frames"] = 99
    assert config["camera_configs"]["camera_1"]["filter"]["min_frames"] == 10
