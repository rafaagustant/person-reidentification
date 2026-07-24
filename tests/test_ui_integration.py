import pandas as pd

from config.cases import DEMO_CASES
from config.presets import CASE_RECOMMENDATIONS, INITIAL_ANALYSIS_CONFIG, build_error_analysis_config
from core.tracking import safe_progress
from ui.config_panel import ALLOWED_TUNING_IMGSZ, _apply_scope, configuration_for_mode
from ui.components import widget_key
from ui.app import validate_reid_inputs
from ui.state import camera_tracking_fingerprint, ensure_camera_tracking_state, refresh_camera_tracking_state, set_reid_failure
from ui.result_adapter import compact_pairs, filtered_tracks_view, format_count, format_ratio, global_id_gallery_view, is_meaningful_reason, merged_global_gallery_view, metric_status, tracking_view
from ui.state import invalidate_for_config_change


def _config(yolo_conf=0.2, threshold=0.8, camera_one=0.2, camera_two=0.3):
    return {
        "tracking": {"yolo_conf": yolo_conf}, "filter": {"min_frames": 2}, "reid": {"cross_threshold": threshold},
        "camera_configs": {"c1": {"tracking": {"yolo_conf": camera_one}, "filter": {}}, "c2": {"tracking": {"yolo_conf": camera_two}, "filter": {}}},
    }


def test_adapter_uses_pipeline_tracking_metrics_key_and_raw_tracks():
    result = {
        "raw_tracks_all": pd.DataFrame({"camera": ["c1", "c2"], "track_id": [1, 1]}),
        "valid_summary_all": pd.DataFrame({"track_key": ["c1_T1"]}),
        "tracking_standard_metrics_df": pd.DataFrame([{ "precision": 0.8, "score_available": True }]),
        "standard_metrics_df": pd.DataFrame([{ "precision": 0.1 }]),
    }
    view = tracking_view(result)
    assert view["raw_tracks"] == 2
    assert view["valid_tracks"] == 1
    assert view["metrics"]["precision"] == 0.8


def test_metric_formatting_has_integer_counts_and_clear_empty_states():
    assert format_count(3.0) == "3"
    assert format_ratio(0.875) == "0.875"
    assert metric_status(None) == "Belum dihitung"
    assert metric_status(None, gt_available=False) == "GT tidak tersedia"
    assert metric_status(None, applicable=False) == "Tidak berlaku"


def test_tracking_change_invalidates_all_downstream_results():
    state = {"tracking_result": {"x": 1}, "reid_result": {"x": 1}, "render_result": {"x": 1}, "tracking_done": True, "reid_done": True, "render_done": True}
    message = invalidate_for_config_change(state, _config(), _config(yolo_conf=0.3))
    assert "Tracking" in message
    assert not state["tracking_done"] and state["tracking_result"] is None and state["render_result"] is None


def test_reid_change_keeps_tracking_and_invalidates_reid():
    state = {"tracking_result": {"x": 1}, "reid_result": {"x": 1}, "render_result": {"x": 1}, "tracking_done": True, "reid_done": True, "render_done": True}
    message = invalidate_for_config_change(state, _config(), _config(threshold=0.9))
    assert "Re-ID" in message
    assert state["tracking_done"] and state["tracking_result"] == {"x": 1}
    assert not state["reid_done"] and state["reid_result"] is None


def test_camera_config_remains_independent_before_explicit_scope_update():
    config = _config()
    assert config["camera_configs"]["c1"]["tracking"]["yolo_conf"] != config["camera_configs"]["c2"]["tracking"]["yolo_conf"]


def test_only_two_active_configuration_modes_and_one_initial_config():
    assert set(INITIAL_ANALYSIS_CONFIG) == {"tracking", "filter", "reid"}
    assert configuration_for_mode("case_1_normal_success", "Konfigurasi Rekomendasi")["config_mode"] == "case_recommendation"
    assert configuration_for_mode("case_1_normal_success", "Konfigurasi Awal Analisis")["config_mode"] == "initial_analysis"


def test_initial_analysis_config_applies_to_all_cases_and_recommendations_remain_per_case():
    configs = [build_error_analysis_config(case["case_id"]) for case in DEMO_CASES]
    assert all(config["camera_configs"] for config in configs)
    assert CASE_RECOMMENDATIONS["case_1_normal_success"] != CASE_RECOMMENDATIONS["case_3_failure_limitation"]


def test_tracking_progress_callback_receives_updates():
    received = []
    safe_progress(lambda camera, done, total: received.append((camera, done, total)), "c1", 5, 10)
    assert received == [("c1", 5, 10)]


def test_widget_keys_are_unique_for_case_camera_and_parameter():
    assert widget_key("case_1", "analysis", "yolo_conf", "camera_1") != widget_key("case_2", "analysis", "yolo_conf", "camera_1")
    assert widget_key("case_1", "analysis", "yolo_conf", "camera_1") != widget_key("case_1", "analysis", "yolo_conf", "camera_2")
    assert widget_key("case_1", "analysis", "scope") != widget_key("case_1", "analysis", "camera")


def test_all_case_tabs_render_without_duplicate_widget_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    messages = "\n".join(str(error) for error in app.exception)
    assert "StreamlitDuplicateElementId" not in messages
    assert "DuplicateWidgetID" not in messages
    assert not app.exception


def test_reid_preflight_rejects_missing_tracking_result(tmp_path):
    valid, message = validate_reid_inputs({"case_id": "case_1"}, {"tracking_done": False}, str(tmp_path / "missing.pth"), False)
    assert not valid
    assert "hasil tracking" in message


def test_reid_preflight_rejects_missing_valid_crop_and_weight(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text('{"case_id": "case_1"}', encoding="utf-8")
    tracks = pd.DataFrame({"track_key": ["c1_T1"], "crop_path": [str(tmp_path / "missing.jpg")]})
    state = {"tracking_done": True, "run_dir": str(run_dir), "tracking_result": {"run_dir": run_dir, "valid_summary_all": pd.DataFrame({"track_key": ["c1_T1"]}), "valid_tracks_all": tracks}, "config": _config()}
    valid, message = validate_reid_inputs({"case_id": "case_1"}, state, str(tmp_path / "missing.pth"), False)
    assert not valid
    assert "Crop valid track" in message


def test_reid_preflight_rejects_tracking_from_different_case(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text('{"case_id": "case_2"}', encoding="utf-8")
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"crop")
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"weight")
    tracks = pd.DataFrame({"track_key": ["c1_T1"], "crop_path": [str(crop)]})
    state = {"tracking_done": True, "run_dir": str(run_dir), "tracking_result": {"run_dir": run_dir, "valid_summary_all": pd.DataFrame({"track_key": ["c1_T1"]}), "valid_tracks_all": tracks}, "config": _config()}
    valid, message = validate_reid_inputs({"case_id": "case_1"}, state, str(weight), False)
    assert not valid
    assert "case yang berbeda" in message


def test_reid_failure_keeps_tracking_and_persists_traceback():
    state = {"tracking_result": {"run": 1}, "reid_result": {"old": 1}, "render_result": {"old": 1}, "tracking_done": True, "reid_done": True, "render_done": True}
    set_reid_failure(state, "crop tidak tersedia", "Traceback: crop tidak tersedia")
    assert state["tracking_result"] == {"run": 1}
    assert state["reid_status"] == "failed"
    assert state["last_error_traceback"] == "Traceback: crop tidak tersedia"
    assert state["reid_result"] is None and state["render_result"] is None


def test_allowed_tuning_imgsz_and_initial_value_are_fixed():
    assert ALLOWED_TUNING_IMGSZ == [640, 960, 1280]
    assert all(isinstance(value, int) for value in ALLOWED_TUNING_IMGSZ)
    assert INITIAL_ANALYSIS_CONFIG["tracking"]["imgsz"] == 960


def test_imgsz_scope_changes_only_selected_camera():
    config = build_error_analysis_config("case_4_multicamera_success")
    cameras = list(config["camera_configs"])
    first, second = cameras[:2]
    original_second = config["camera_configs"][second]["tracking"]["imgsz"]
    tracking = dict(config["camera_configs"][first]["tracking"], imgsz=1280)
    updated = _apply_scope(config, tracking, config["camera_configs"][first]["filter"], config["reid"], "Kamera tertentu", first)
    assert updated["camera_configs"][first]["tracking"]["imgsz"] == 1280
    assert updated["camera_configs"][second]["tracking"]["imgsz"] == original_second


def test_adapter_reads_legacy_mota_fields_without_exposing_them():
    base = {"raw_tracks_all": pd.DataFrame(), "valid_summary_all": pd.DataFrame(), "valid_tracks_all": pd.DataFrame(), "tracking_standard_metrics_df": pd.DataFrame([{"score_available": True, "raw_mota_simple": -0.5}])}
    assert tracking_view(base)["metrics"]["mota"] == -0.5
    base["tracking_standard_metrics_df"] = pd.DataFrame([{"score_available": True, "mota_simple": -0.25}])
    assert tracking_view(base)["metrics"]["mota"] == -0.25


def test_filtered_track_gallery_metadata_uses_only_invalid_tracks_and_sorts_closest_first():
    summary = pd.DataFrame([
        {"track_key": "c1_T1", "camera": "c1", "track_id": 1, "is_valid": True, "num_frames": 30},
        {"track_key": "c1_T2", "camera": "c1", "track_id": 2, "is_valid": False, "filter_reason": "min_frames", "num_frames": 14, "min_frames_used": 15},
        {"track_key": "c1_T3", "camera": "c1", "track_id": 3, "is_valid": False, "filter_reason": "min_frames", "num_frames": 2, "min_frames_used": 15},
    ])
    filtered = filtered_tracks_view(summary)
    assert list(filtered["track_key"]) == ["c1_T2", "c1_T3"]
    assert not filtered["is_valid"].any()


def test_compact_pair_tables_hide_placeholder_reasons_and_keep_real_rejection_reason():
    pairs = pd.DataFrame([
        {"track_a": "c1_T1", "track_b": "c2_T2", "cosine_similarity": 0.9, "merge_status": True, "merge_reason": "cross_camera_mnn"},
        {"track_a": "c1_T3", "track_b": "c2_T4", "cosine_similarity": 0.7, "merge_status": False, "merge_reason": "not merged"},
        {"track_a": "c1_T5", "track_b": "c2_T6", "cosine_similarity": 0.6, "merge_status": False, "merge_reason": "below_threshold"},
    ])
    compact = compact_pairs(pairs, 0.8)
    assert list(compact["decision"]) == ["Diterima", "Ditolak", "Ditolak"]
    assert compact.loc[0, "rejection_reason"] == ""
    assert compact.loc[1, "rejection_reason"] == ""
    assert compact.loc[2, "rejection_reason"] == "below_threshold"


def test_placeholder_reason_values_are_not_meaningful():
    for value in ["not merged", "NOT_MERGED", "none", "N/A", "", None, "accepted"]:
        assert not is_meaningful_reason(value)
    assert is_meaningful_reason("temporal_overlap")


def test_merged_track_gallery_groups_final_global_ids_and_excludes_singletons():
    meta = pd.DataFrame([
        {"global_id": 4, "track_key": "camera_3_T7", "camera": "camera_3", "track_id": 7, "representative_crop": "three.jpg"},
        {"global_id": 1, "track_key": "camera_1_T1", "camera": "camera_1", "track_id": 1, "representative_crop": "one.jpg"},
        {"global_id": 4, "track_key": "camera_1_T3", "camera": "camera_1", "track_id": 3, "representative_crop": "two.jpg"},
        {"global_id": 4, "track_key": "camera_2_T1", "camera": "camera_2", "track_id": 1, "representative_crop": "four.jpg"},
    ])
    groups, members = merged_global_gallery_view(meta)
    assert list(groups["global_id"]) == [4]
    assert groups.iloc[0]["num_member_tracks"] == 3
    assert list(members[members["global_id"] == 4]["track_key"]) == ["camera_1_T3", "camera_2_T1", "camera_3_T7"]


def test_merged_track_gallery_preserves_all_members_beyond_four():
    meta = pd.DataFrame([
        {"global_id": 10, "track_key": f"camera_{camera}_T{track}", "camera": f"camera_{camera}", "track_id": track, "crop_path": f"{track}.jpg"}
        for camera, track in [(3, 7), (1, 8), (2, 4), (1, 2), (3, 1), (2, 9)]
    ])
    groups, members = merged_global_gallery_view(meta)
    assert groups.iloc[0]["num_member_tracks"] == 6
    assert len(members[members["global_id"] == 10]) == 6
    assert list(members[members["global_id"] == 10]["track_key"]) == [
        "camera_1_T2", "camera_1_T8", "camera_2_T4", "camera_2_T9", "camera_3_T1", "camera_3_T7",
    ]


def test_global_id_gallery_uses_one_deterministic_crop_and_lists_all_members():
    meta = pd.DataFrame([
        {"global_id": 4, "track_key": "camera_2_T1", "camera": "camera_2", "track_id": 1, "representative_crop": "b.jpg", "avg_area": 100, "avg_conf": 0.8},
        {"global_id": 4, "track_key": "camera_1_T3", "camera": "camera_1", "track_id": 3, "representative_crop": "a.jpg", "avg_area": 100, "avg_conf": 0.9},
        {"global_id": 2, "track_key": "camera_1_T5", "camera": "camera_1", "track_id": 5, "representative_crop": "single.jpg", "avg_area": 50, "avg_conf": 0.7},
        {"global_id": 4, "track_key": "camera_3_T7", "camera": "camera_3", "track_id": 7, "representative_crop": "c.jpg", "avg_area": 90, "avg_conf": 0.9},
    ])
    cards = global_id_gallery_view(meta, pd.DataFrame())
    assert list(cards["global_id"]) == [2, 4]
    merged = cards[cards["global_id"] == 4].iloc[0]
    assert merged["representative_track_key"] == "camera_1_T3"
    assert merged["representative_crop_path"] == "a.jpg"
    assert merged["num_member_tracks"] == 3
    assert merged["member_track_keys"] == "camera_1_T3, camera_2_T1, camera_3_T7"
    assert cards[cards["global_id"] == 2].iloc[0]["num_member_tracks"] == 1


def test_pair_adapter_exposes_temporal_metadata_and_canonical_mnn_labels():
    pairs = pd.DataFrame([
        {"track_a": "a", "track_b": "b", "camera_a": "c1", "camera_b": "c2", "same_camera": False, "cosine_similarity": 0.9, "merge_status": True, "mnn_status_code": "passed", "overlap_frame_count": 1, "temporal_gap": 0, "temporal_status": "Overlap"},
        {"track_a": "a", "track_b": "c", "camera_a": "c1", "camera_b": "c2", "same_camera": False, "cosine_similarity": 0.7, "merge_status": False, "mnn_status_code": "failed", "overlap_frame_count": 0, "temporal_gap": 3, "temporal_status": "Terpisah"},
        {"track_a": "b", "track_b": "c", "camera_a": "c2", "camera_b": "c2", "same_camera": True, "cosine_similarity": 0.8, "merge_status": False, "mnn_status_code": "not_applicable"},
        {"track_a": "d", "track_b": "e", "camera_a": "c3", "camera_b": "c4", "same_camera": False, "cosine_similarity": 0.6, "merge_status": False, "mnn_status_code": "not_evaluated"},
    ])
    compact = compact_pairs(pairs, 0.8)
    assert list(compact["mnn_status"]) == ["Diterima", "Ditolak", "Tidak berlaku", "Tidak dievaluasi"]
    assert compact.loc[0, "overlap_frame_count"] == 1
    assert compact.loc[1, "temporal_status"] == "Terpisah"
    assert "overlap_ratio" not in compact


def test_pair_adapter_uses_compact_frame_ranges_and_legacy_fallbacks():
    pairs = pd.DataFrame([
        {"track_a": "a", "track_b": "b", "cosine_similarity": 0.9, "start_source_frame_a": 1201, "end_source_frame_a": 1450, "start_source_frame_b": 1380, "end_source_frame_b": 1380, "temporal_status": "Overlap", "overlap_frame_count": 1, "temporal_gap": 0},
        {"track_a": "c", "track_b": "d", "cosine_similarity": 0.8, "first_frame_a": 3, "last_frame_a": 5, "first_frame_b": 8, "last_frame_b": 9},
    ])
    compact = compact_pairs(pairs, 0.8)
    assert list(compact["frame_range_a"]) == ["1201–1450", "3–5"]
    assert list(compact["frame_range_b"]) == ["1380", "8–9"]
    assert compact.loc[1, "temporal_status"] == "Tidak tersedia"
    assert pd.isna(compact.loc[1, "overlap_frame_count"])


def test_camera_tracking_state_marks_only_changed_camera_stale():
    case = {"case_id": "case_x", "cameras": ["camera_1", "camera_2", "camera_3"], "video_files": {"camera_1": "a.mp4", "camera_2": "b.mp4", "camera_3": "c.mp4"}}
    config = _config()
    config["camera_configs"] = {
        "camera_1": {"tracking": {"yolo_conf": 0.2}, "filter": {}},
        "camera_2": {"tracking": {"yolo_conf": 0.3}, "filter": {}},
        "camera_3": {"tracking": {"yolo_conf": 0.4}, "filter": {}},
    }
    state = {"config": config, "camera_tracking_status": {}, "camera_tracking_results": {}}
    ensure_camera_tracking_state(case, state)
    for camera in case["cameras"]:
        state["camera_tracking_status"][camera] = "ready"
        state["camera_tracking_config_fingerprints"][camera] = camera_tracking_fingerprint(case, config, camera)
    changed = _config()
    changed["camera_configs"] = {key: {"tracking": dict(value["tracking"]), "filter": {}} for key, value in config["camera_configs"].items()}
    changed["camera_configs"]["camera_2"]["tracking"]["yolo_conf"] = 0.9
    state["config"] = changed
    refresh_camera_tracking_state(case, state)
    assert state["camera_tracking_status"] == {"camera_1": "ready", "camera_2": "stale", "camera_3": "ready"}
    assert state["has_tracking_results"] and not state["tracking_done"]


def test_yolo_iou_scope_preserves_other_cameras_until_submit():
    config = _config()
    config["camera_configs"]["c1"]["tracking"]["yolo_iou"] = 0.50
    config["camera_configs"]["c2"]["tracking"]["yolo_iou"] = 0.55
    config["camera_configs"]["c2"]["tracking"]["match_thresh"] = 0.80
    pending_tracking = dict(config["camera_configs"]["c2"]["tracking"], yolo_iou=0.60)
    assert config["camera_configs"]["c2"]["tracking"]["yolo_iou"] == 0.55
    updated = _apply_scope(config, pending_tracking, config["camera_configs"]["c2"]["filter"], config["reid"], "Kamera tertentu", "c2")
    assert updated["camera_configs"]["c1"]["tracking"]["yolo_iou"] == 0.50
    assert updated["camera_configs"]["c2"]["tracking"]["yolo_iou"] == 0.60
    assert updated["camera_configs"]["c2"]["tracking"]["match_thresh"] == config["camera_configs"]["c2"]["tracking"]["match_thresh"]
