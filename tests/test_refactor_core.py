import json

import numpy as np
import pandas as pd

from config.presets import build_config_from_case_recommendation, normalize_config
from core.association import assign_global_ids, mutual_nearest_cross_pairs
from core.evaluation import build_tracking_standard_metrics, match_predictions_to_gt
from core.reid import build_embedding_manifest, build_sampled_track_crop_df, embedding_cache_is_valid
from core.tracking import make_botsort_yaml


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
    assert metric["mota_simple"] < 0


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
