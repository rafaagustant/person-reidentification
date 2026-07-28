import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from config.presets import build_recommended_config, get_camera_stage_config, normalize_config
from core.association import assign_global_ids, compute_track_similarity_df, mutual_nearest_cross_pairs
from core.evaluation import build_gt_temporal_coverage, build_track_gt_diagnostics, build_tracking_standard_metrics, match_predictions_to_gt
from core.reid import build_embedding_manifest, build_sampled_track_crop_df, embedding_cache_is_valid
from core.tracking import build_track_count_summary, make_botsort_yaml, run_track_once, save_tracking_runtime_manifest
from core.pipeline import _camera_status_rows, _evaluate_tracking_scope


def _rows(rows):
    return pd.DataFrame(rows, columns=["camera", "source_frame", "x1", "y1", "x2", "y2"])


def test_gt_matching_is_one_to_one_and_deterministic():
    predictions = _rows([["c1", 1, 0, 0, 10, 10], ["c1", 1, 0, 0, 10, 10]])
    ground_truth = _rows([["c1", 1, 0, 0, 10, 10]])
    ground_truth["gt_id"] = [7]
    matched = match_predictions_to_gt(predictions, ground_truth)
    assert matched["is_matched"].sum() == 1
    assert list(matched["gt_id"]) == [7, -1]


def test_frame_and_camera_do_not_cross_match():
    predictions = _rows([["c1", 1, 0, 0, 10, 10], ["c2", 2, 0, 0, 10, 10]])
    ground_truth = _rows([["c1", 2, 0, 0, 10, 10]])
    ground_truth["gt_id"] = [1]
    assert not match_predictions_to_gt(predictions, ground_truth)["is_matched"].any()


def test_track_gt_diagnostics_uses_source_frame_range_and_continuity():
    predictions = _rows([
        ["c1", 100, 0, 0, 10, 10], ["c1", 101, 0, 0, 10, 10], ["c1", 103, 0, 0, 10, 10],
    ])
    predictions["track_id"] = 10
    predictions["track_key"] = "c1_T10"
    ground_truth = _rows([["c1", 100, 0, 0, 10, 10], ["c1", 101, 0, 0, 10, 10], ["c1", 103, 0, 0, 10, 10]])
    ground_truth["gt_id"] = 6
    matched = match_predictions_to_gt(predictions, ground_truth)
    summary = pd.DataFrame([{"camera": "c1", "track_id": 10, "track_key": "c1_T10", "num_frames": 3, "first_source_frame": 100, "last_source_frame": 103, "observed_frame_count": 3, "continuity_ratio": 0.75}])
    diagnostic = build_track_gt_diagnostics(matched, summary, 0.5).iloc[0]
    assert (diagnostic["first_source_frame"], diagnostic["last_source_frame"], diagnostic["num_frames"]) == (100, 103, 3)
    assert diagnostic["continuity_ratio"] == 0.75
    assert diagnostic["matched_frame_count"] == 3
    assert diagnostic["unmatched_frame_count"] == 0


def test_match_diagnosis_classifies_no_gt_below_iou_and_claimed_prediction():
    predictions = _rows([
        ["c1", 1, 0, 0, 10, 10], ["c1", 2, 0, 0, 10, 10],
        ["c1", 3, 0, 0, 10, 10], ["c1", 3, 0, 0, 10, 10],
    ])
    predictions["track_id"] = [1, 2, 3, 4]
    predictions["track_key"] = ["c1_T1", "c1_T2", "c1_T3", "c1_T4"]
    gt = _rows([["c1", 2, 20, 20, 30, 30], ["c1", 3, 0, 0, 10, 10]])
    gt["gt_id"] = [6, 7]
    matched = match_predictions_to_gt(predictions, gt)
    by_track = matched.set_index("track_key")
    assert by_track.loc["c1_T1", "match_status"] == "no_gt_on_frame"
    assert by_track.loc["c1_T2", "match_status"] == "below_iou_threshold"
    assert by_track.loc["c1_T3", "match_status"] == "matched"
    assert by_track.loc["c1_T4", "match_status"] == "gt_claimed_by_other_prediction"
    assert by_track.loc["c1_T4", "claimed_by_track_key"] == "c1_T3"


def test_track_diagnosis_totals_match_tp_and_fp_and_filtered_track_is_not_evaluated():
    predictions = _rows([["c1", 1, 0, 0, 10, 10], ["c1", 1, 0, 0, 10, 10]])
    predictions["track_id"] = [1, 2]
    predictions["track_key"] = ["c1_T1", "c1_T2"]
    gt = _rows([["c1", 1, 0, 0, 10, 10]])
    gt["gt_id"] = [7]
    matched = match_predictions_to_gt(predictions, gt)
    summary = pd.DataFrame([
        {"camera": "c1", "track_id": 1, "track_key": "c1_T1", "num_frames": 1},
        {"camera": "c1", "track_id": 2, "track_key": "c1_T2", "num_frames": 1},
        {"camera": "c1", "track_id": 9, "track_key": "c1_T9", "num_frames": 5},
    ])
    diagnostic = build_track_gt_diagnostics(matched, summary, 0.5)
    metrics = build_tracking_standard_metrics(matched, gt).iloc[0]
    assert diagnostic["matched_frame_count"].sum() == metrics["tp"]
    assert diagnostic["unmatched_frame_count"].sum() == metrics["fp"]
    filtered = diagnostic.set_index("track_key").loc["c1_T9"]
    assert not filtered["evaluation_included"]
    assert filtered["matched_frame_count"] == 0


def test_mota_is_not_clipped_when_errors_exceed_gt():
    matched = pd.DataFrame({"gt_id": [-1, -1], "camera": ["c", "c"], "source_frame": [1, 1]})
    gt = pd.DataFrame({"gt_id": [1], "camera": ["c"], "source_frame": [1]})
    metric = build_tracking_standard_metrics(matched, gt).iloc[0]
    assert metric["mota"] < 0


def test_temporal_coverage_uses_annotated_source_frames_and_track_keys():
    gt = pd.DataFrame({"camera": ["c1", "c1", "c1", "c1", "c2"], "gt_id": [1, 1, 1, 1, 1], "source_frame": [1, 2, 4, 5, 1]})
    matched = pd.DataFrame({"camera": ["c1", "c1"], "gt_id": [1, 1], "source_frame": [1, 5], "track_key": ["c1_T2", "c1_T1"]})
    coverage, summary = build_gt_temporal_coverage(matched, gt)
    row = coverage[(coverage["camera"] == "c1") & (coverage["gt_id"] == 1)].iloc[0]
    assert row["gt_frame_count"] == 4
    assert row["matched_frame_count"] == 2
    assert row["missed_frame_count"] == 2
    assert row["temporal_coverage"] == 0.5
    assert row["longest_missed_gap"] == 1
    assert row["matched_track_keys"] == "c1_T1, c1_T2"
    assert summary.iloc[0]["total_gt_frame_count"] == 5


def test_sampling_uses_each_camera_filter():
    valid = pd.DataFrame({
        "track_key": ["c1_T1"] * 4 + ["c2_T2"] * 4,
        "camera": ["c1"] * 4 + ["c2"] * 4,
        "frame": list(range(4)) * 2, "conf": [0.8] * 8, "area": [100] * 8,
    })
    sampled = build_sampled_track_crop_df(valid, {"max_samples_per_track": 4}, {
        "c1": {"filter": {"max_samples_per_track": 1}},
        "c2": {"filter": {"max_samples_per_track": 2}},
    })
    assert sampled.groupby("camera").size().to_dict() == {"c1": 1, "c2": 2}


def _track_rows(camera: str, track_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"camera": camera, "track_id": track_id, "frame": frame}
         for track_id in track_ids for frame in (1, 2)]
    )


def test_track_counts_distinguish_raw_valid_and_filtered_tracks():
    raw = _track_rows("camera_1", [1, 2, 3])
    valid = _track_rows("camera_1", [1, 2])
    counts = build_track_count_summary(raw, valid, ["camera_1"])
    assert counts["raw_track_count"] == 3
    assert counts["valid_track_count"] == 2
    assert counts["filtered_track_count"] == 1
    assert counts["by_camera_df"].iloc[0].to_dict() == {
        "camera": "camera_1",
        "raw_track_count": 3,
        "valid_track_count": 2,
        "filtered_track_count": 1,
    }


def test_track_counts_use_camera_and_track_id_for_multicamera_totals():
    raw = pd.concat(
        [_track_rows("camera_1", [1, 2]), _track_rows("camera_2", [1, 2, 3])],
        ignore_index=True,
    )
    counts = build_track_count_summary(raw, raw.copy(), ["camera_1", "camera_2"])
    assert counts["raw_track_count"] == 5
    assert counts["valid_track_count"] == 5
    assert counts["filtered_track_count"] == 0


@pytest.mark.parametrize(
    "raw_ids,valid_ids,expected",
    [
        ([1, 2], [1, 2], (2, 2, 0)),
        ([1, 2], [], (2, 0, 2)),
        ([], [], (0, 0, 0)),
    ],
)
def test_track_count_edge_cases(raw_ids, valid_ids, expected):
    counts = build_track_count_summary(
        _track_rows("camera_1", raw_ids),
        _track_rows("camera_1", valid_ids),
        ["camera_1"],
    )
    assert (
        counts["raw_track_count"],
        counts["valid_track_count"],
        counts["filtered_track_count"],
    ) == expected


def test_mnn_is_per_camera_pair_and_union_is_transitive():
    pairs = pd.DataFrame([
        ["a", "c1", 1, 2, "b", "c2", 1, 2, False, 0, 0, 0.9],
        ["a", "c1", 1, 2, "c", "c3", 1, 2, False, 0, 0, 0.8],
        ["b", "c2", 1, 2, "c", "c3", 1, 2, False, 0, 0, 0.7],
    ], columns=["track_a", "camera_a", "first_frame_a", "last_frame_a", "track_b", "camera_b", "first_frame_b", "last_frame_b", "same_camera", "temporal_gap", "temporal_overlap", "cosine_similarity"])
    assert mutual_nearest_cross_pairs(pairs, 0.75) == {("a", "b"), ("a", "c")}
    tracks = pd.DataFrame({"track_key": ["a", "b", "c"], "camera": ["c1", "c2", "c3"], "first_frame": [1, 1, 1], "last_frame": [2, 2, 2]})
    global_meta, _ = assign_global_ids(tracks, pairs, {"enable_cross_camera": True, "use_mnn": True, "cross_threshold": 0.75, "enable_strict_intra": False})
    assert global_meta["global_id"].nunique() == 1


def test_pair_temporal_metrics_use_actual_source_frame_sets():
    tracks = pd.DataFrame([
        {"track_key": "a", "camera": "c1", "first_frame": 0, "last_frame": 5, "source_frames": (100, 101, 105)},
        {"track_key": "b", "camera": "c2", "first_frame": 0, "last_frame": 5, "source_frames": (102, 103, 105)},
    ])
    pair = compute_track_similarity_df(tracks, np.array([[1.0, 0.0], [0.0, 1.0]])).iloc[0]
    assert pair["start_source_frame_a"] == 100
    assert pair["end_source_frame_a"] == 105
    assert pair["overlap_frame_count"] == 1
    assert "overlap_ratio" not in pair.index
    assert pair["temporal_gap"] == 0
    assert pair["temporal_status"] == "Overlap"


def test_pair_temporal_gap_and_status_handle_ordered_and_missing_metadata():
    tracks = pd.DataFrame([
        {"track_key": "a", "camera": "c1", "first_frame": 1, "last_frame": 2, "source_frames": (100,)},
        {"track_key": "b", "camera": "c2", "first_frame": 3, "last_frame": 4, "source_frames": (101,)},
        {"track_key": "c", "camera": "c3", "first_frame": 8, "last_frame": 9, "source_frames": (105,)},
        {"track_key": "d", "camera": "c4", "first_frame": 1, "last_frame": 2},
    ])
    pairs = compute_track_similarity_df(tracks, np.eye(4))
    ordered = pairs[(pairs["track_a"] == "a") & (pairs["track_b"] == "b")].iloc[0]
    separated = pairs[(pairs["track_a"] == "a") & (pairs["track_b"] == "c")].iloc[0]
    unavailable = pairs[(pairs["track_a"] == "a") & (pairs["track_b"] == "d")].iloc[0]
    assert (ordered["temporal_gap"], ordered["temporal_status"]) == (0, "Berurutan")
    assert (separated["temporal_gap"], separated["temporal_status"]) == (4, "Terpisah")
    assert unavailable["temporal_status"] == "Tidak tersedia"


def test_mnn_status_metadata_does_not_change_association_decisions():
    tracks = pd.DataFrame({"track_key": ["a", "b", "c"], "camera": ["c1", "c2", "c2"], "first_frame": [1, 1, 1], "last_frame": [2, 2, 2]})
    pairs = pd.DataFrame([
        {"track_a": "a", "track_b": "b", "camera_a": "c1", "camera_b": "c2", "same_camera": False, "temporal_gap": 0, "temporal_overlap": 0, "cosine_similarity": 0.9},
        {"track_a": "a", "track_b": "c", "camera_a": "c1", "camera_b": "c2", "same_camera": False, "temporal_gap": 0, "temporal_overlap": 0, "cosine_similarity": 0.8},
        {"track_a": "b", "track_b": "c", "camera_a": "c2", "camera_b": "c2", "same_camera": True, "temporal_gap": 0, "temporal_overlap": 0, "cosine_similarity": 0.95},
    ])
    _, result = assign_global_ids(tracks, pairs, {"enable_cross_camera": True, "use_mnn": True, "cross_threshold": 0.75, "enable_strict_intra": False})
    by_pair = result.set_index(["track_a", "track_b"])
    assert bool(by_pair.loc[("a", "b"), "merge_status"])
    assert not bool(by_pair.loc[("a", "c"), "merge_status"])
    assert by_pair.loc[("a", "b"), "mnn_status_code"] == "passed"
    assert by_pair.loc[("a", "c"), "mnn_status_code"] == "failed"
    assert by_pair.loc[("b", "c"), "mnn_status_code"] == "not_applicable"


def _single_camera_pair(similarity=0.80, gap=0, overlap=0):
    first_b = 101 - overlap if overlap else 101 + gap
    tracks = pd.DataFrame([
        {"track_key": "a", "camera": "c1", "first_frame": 1, "last_frame": 100},
        {"track_key": "b", "camera": "c1", "first_frame": first_b, "last_frame": first_b + 99},
    ])
    pairs = pd.DataFrame([{
        "track_a": "a", "track_b": "b", "camera_a": "c1", "camera_b": "c1",
        "same_camera": True, "cosine_similarity": similarity,
        "temporal_gap": gap, "temporal_overlap": overlap,
        "association_temporal_gap": gap, "association_temporal_overlap": overlap,
    }])
    return tracks, pairs


def test_intra_pair_reports_effective_threshold_and_signed_margin():
    tracks, pairs = _single_camera_pair(similarity=0.79)
    _, result = assign_global_ids(tracks, pairs, {"enable_strict_intra": True, "intra_threshold": 0.76, "cross_threshold": 0.0, "intra_max_gap": 30, "intra_max_overlap": 0})
    row = result.iloc[0]
    assert row["threshold"] == 0.76
    assert row["similarity_margin"] == pytest.approx(0.03)
    assert row["association_type"] == "intra_camera"


@pytest.mark.parametrize("overlap,accepted", [(51, True), (61, False)])
def test_strict_intra_respects_configured_overlap(overlap, accepted):
    tracks, pairs = _single_camera_pair(overlap=overlap)
    _, result = assign_global_ids(tracks, pairs, {"enable_strict_intra": True, "intra_threshold": 0.76, "intra_max_gap": 300, "intra_max_overlap": 60})
    assert bool(result.iloc[0]["merge_status"]) is accepted
    if not accepted:
        assert "rejected_temporal_overlap" in result.iloc[0]["rejection_reason"]


def test_strict_intra_uses_actual_source_frame_overlap():
    tracks, pairs = _single_camera_pair(overlap=118)
    tracks["source_frames"] = [
        tuple(range(1, 101)),
        tuple(range(50, 101)) + tuple(range(150, 199)),
    ]
    pairs["overlap_frame_count"] = 51
    pairs["temporal_gap"] = 0

    _, result = assign_global_ids(
        tracks,
        pairs,
        {
            "enable_strict_intra": True,
            "intra_threshold": 0.76,
            "intra_max_gap": 300,
            "intra_max_overlap": 60,
        },
    )

    assert bool(result.iloc[0]["merge_status"])


@pytest.mark.parametrize("gap,accepted", [(300, True), (301, False)])
def test_strict_intra_respects_configured_gap(gap, accepted):
    tracks, pairs = _single_camera_pair(gap=gap)
    _, result = assign_global_ids(tracks, pairs, {"enable_strict_intra": True, "intra_threshold": 0.76, "intra_max_gap": 300, "intra_max_overlap": 60})
    assert bool(result.iloc[0]["merge_status"]) is accepted
    if not accepted:
        assert "rejected_temporal_gap" in result.iloc[0]["rejection_reason"]


def test_every_rejected_pair_has_reason():
    tracks, pairs = _single_camera_pair(similarity=0.10)
    _, result = assign_global_ids(tracks, pairs, {"enable_strict_intra": True, "intra_threshold": 0.76, "intra_max_gap": 30, "intra_max_overlap": 0})
    assert result.iloc[0]["rejection_reason"] == "rejected_similarity"


def test_embedding_manifest_rejects_changed_crops(tmp_path):
    valid = pd.DataFrame({"track_key": ["a"], "camera": ["c1"], "frame": [1], "crop_path": ["a.jpg"]})
    sampled = valid.copy()
    features_path = tmp_path / "features.npy"
    np.save(features_path, np.ones((1, 4), dtype=np.float32))
    manifest = build_embedding_manifest(valid, sampled, None, 4)
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert embedding_cache_is_valid(manifest_path, build_embedding_manifest(valid, sampled, None), features_path)
    changed = sampled.assign(crop_path="different.jpg")
    assert not embedding_cache_is_valid(manifest_path, build_embedding_manifest(valid, changed, None), features_path)


def test_embedding_manifest_rejects_changed_sampling_config(tmp_path):
    valid = pd.DataFrame(
        {
            "track_key": ["a", "a"],
            "camera": ["c1", "c1"],
            "frame": [1, 2],
            "crop_path": ["a1.jpg", "a2.jpg"],
            "conf": [0.9, 0.8],
            "area": [1000.0, 900.0],
        }
    )
    sampled_two = build_sampled_track_crop_df(valid, {"max_samples_per_track": 2})
    sampled_one = build_sampled_track_crop_df(valid, {"max_samples_per_track": 1})
    features_path = tmp_path / "features.npy"
    np.save(features_path, np.ones((1, 4), dtype=np.float32))
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(build_embedding_manifest(valid, sampled_two, None, 4)),
        encoding="utf-8",
    )

    assert not embedding_cache_is_valid(
        manifest_path,
        build_embedding_manifest(valid, sampled_one, None),
        features_path,
    )


def test_config_validation_and_tracker_yaml(tmp_path):
    config = build_recommended_config("case_1_normal_success")
    config["tracking"]["track_low_thresh"] = 0.9
    config["tracking"]["track_high_thresh"] = 0.1
    try:
        normalize_config(config)
        assert False, "expected invalid thresholds"
    except ValueError:
        pass
    yaml_path = make_botsort_yaml(build_recommended_config("case_1_normal_success")["tracking"], tmp_path)
    assert "with_reid: false" in yaml_path.read_text(encoding="utf-8")


def test_yolo_iou_is_sent_to_ultralytics_track_without_changing_match_threshold(tmp_path):
    class Model:
        def __init__(self):
            self.kwargs = None

        def track(self, frame, **kwargs):
            self.kwargs = kwargs
            return []

    model = Model()
    config = {"yolo_conf": 0.1, "yolo_iou": 0.60, "imgsz": 960, "match_thresh": 0.81}
    run_track_once(model, "frame", tmp_path / "tracker.yaml", config, "cpu")
    assert model.kwargs["iou"] == 0.60
    assert "yolo_iou" not in model.kwargs
    assert config["match_thresh"] == 0.81


def test_yolo_iou_validation_rejects_zero_without_changing_defaults():
    config = build_recommended_config("case_1_normal_success")
    config["tracking"]["yolo_iou"] = 0.0
    try:
        normalize_config(config)
        assert False, "expected invalid yolo_iou"
    except ValueError as error:
        assert "yolo_iou" in str(error)


def test_custom_camera_config_reaches_runtime_without_legacy_preset_override(tmp_path):
    class Model:
        def __init__(self):
            self.kwargs = None

        def track(self, frame, **kwargs):
            self.kwargs = kwargs
            return []

    config = normalize_config({
        "config_mode": "initial_analysis",
        "tracking": {"yolo_conf": 0.11, "yolo_iou": 0.51, "imgsz": 640, "track_buffer": 31},
        "filter": {"min_frames": 12, "max_samples_per_track": 17},
        "reid": {"cross_threshold": 0.72, "intra_threshold": 0.70},
        "camera_configs": {
            "camera_1": {"tracking": {"yolo_conf": 0.21, "yolo_iou": 0.61, "imgsz": 960, "track_buffer": 41}, "filter": {"min_frames": 22}},
            "camera_2": {"tracking": {"yolo_conf": 0.31, "yolo_iou": 0.71, "imgsz": 1280, "track_buffer": 51}, "filter": {"min_frames": 32}},
        },
    })
    tracking_one, filtering_one = get_camera_stage_config(config, "camera_1")
    tracking_two, filtering_two = get_camera_stage_config(config, "camera_2")
    assert tracking_one["yolo_iou"] == 0.61
    assert tracking_two["yolo_iou"] == 0.71
    assert filtering_one["min_frames"] == 22
    assert filtering_two["min_frames"] == 32
    assert config["reid"]["cross_threshold"] == 0.72

    model = Model()
    run_track_once(model, "frame", tmp_path / "tracker.yaml", tracking_two, "cpu")
    assert model.kwargs["conf"] == 0.31
    assert model.kwargs["iou"] == 0.71
    assert model.kwargs["imgsz"] == 1280
    assert tracking_two["match_thresh"] == config["camera_configs"]["camera_2"]["tracking"]["match_thresh"]

    runtime_manifest = save_tracking_runtime_manifest(tracking_two, tmp_path / "tracker.yaml", tmp_path)
    manifest = yaml.safe_load(runtime_manifest.read_text(encoding="utf-8"))
    assert manifest["tracking"]["yolo_iou"] == 0.71
    assert "tracking_config" not in manifest


def test_tracking_evaluation_scope_keeps_academic_outputs_without_diagnostics_or_scores(tmp_path):
    predictions = pd.DataFrame([{
        "camera": "camera_1", "frame": 1, "source_frame": 101, "track_id": 3,
        "track_key": "camera_1_T3", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
    }])
    ground_truth = pd.DataFrame([{
        "camera": "camera_1", "source_frame": 1, "gt_id": 6,
        "x1": 0, "y1": 0, "x2": 10, "y2": 10,
    }])
    result = _evaluate_tracking_scope(
        {"case_id": "case_test", "cameras": ["camera_1"]},
        predictions,
        ground_truth,
        tmp_path,
        include_temporal_coverage=True,
    )
    assert result["tracking_standard_metrics_df"].iloc[0]["tp"] == 1
    assert result["gt_temporal_coverage_df"].iloc[0]["matched_frame_count"] == 1
    assert (tmp_path / "tracking_standard_metrics.csv").exists()
    assert (tmp_path / "gt_temporal_coverage.csv").exists()
    assert (tmp_path / "track_gt_diagnostics.csv").exists()
    assert not any((tmp_path / name).exists() for name in [
        "tracking_diagnostics.csv", "tracking_tuning_hints.csv", "tuning_hints.json", "flow1_tracking_score.csv",
    ])


def test_camera_status_is_objective_without_composite_tracking_score(tmp_path):
    camera_dir = tmp_path / "camera_1"
    camera_dir.mkdir()
    pd.DataFrame({"track_key": ["camera_1_T1"]}).to_csv(camera_dir / "local_tracks_valid.csv", index=False)
    status = _camera_status_rows(
        {"cameras": ["camera_1"]},
        tmp_path,
        {"camera_1"},
        {"camera_1": 1.25},
    )
    assert status.iloc[0]["status"] == "done"
    assert status.iloc[0]["num_valid_tracks"] == 1
    assert status.iloc[0]["runtime_sec"] == 1.25
    assert "tracking_score" not in status.columns


def test_removed_diagnostics_and_scoring_modules_have_no_production_imports():
    assert not Path("core/diagnostics.py").exists()
    assert not Path("config/scoring.py").exists()
    source = Path("core/pipeline.py").read_text(encoding="utf-8")
    assert "core.diagnostics" not in source
    assert "config.scoring" not in source
