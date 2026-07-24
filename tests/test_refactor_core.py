import json

import numpy as np
import pandas as pd

from config.presets import build_config_from_case_recommendation, normalize_config
from core.association import assign_global_ids, compute_track_similarity_df, mutual_nearest_cross_pairs
from core.evaluation import build_gt_temporal_coverage, build_tracking_standard_metrics, match_predictions_to_gt
from core.reid import build_embedding_manifest, build_sampled_track_crop_df, embedding_cache_is_valid
from core.tracking import make_botsort_yaml, run_track_once


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
        "c1": {"filter": {"max_samples_per_track": 1, "crop_selection_strategy": "uniform"}},
        "c2": {"filter": {"max_samples_per_track": 2, "crop_selection_strategy": "uniform"}},
    })
    assert sampled.groupby("camera").size().to_dict() == {"c1": 1, "c2": 2}


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


def test_config_validation_and_tracker_yaml(tmp_path):
    config = build_config_from_case_recommendation("case_1_normal_success")
    config["tracking"]["track_low_thresh"] = 0.9
    config["tracking"]["track_high_thresh"] = 0.1
    try:
        normalize_config(config)
        assert False, "expected invalid thresholds"
    except ValueError:
        pass
    yaml_path = make_botsort_yaml(build_config_from_case_recommendation("case_1_normal_success")["tracking"], tmp_path)
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
    config = build_config_from_case_recommendation("case_1_normal_success")
    config["tracking"]["yolo_iou"] = 0.0
    try:
        normalize_config(config)
        assert False, "expected invalid yolo_iou"
    except ValueError as error:
        assert "yolo_iou" in str(error)
