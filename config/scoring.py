from __future__ import annotations


SCORE_WEIGHTS = {
    "tracking": {
        "identity_coverage_rate": 0.20,
        "precision": 0.15,
        "recall": 0.15,
        "f1": 0.20,
        "mean_track_purity": 0.15,
        "fragmentation_quality": 0.05,
        "false_positive_quality": 0.05,
        "mixed_track_quality": 0.05,
    },
    "reid": {
        "mean_global_id_purity": 0.40,
        "mixed_gid_quality": 0.25,
        "fragment_reduction_quality": 0.25,
        "merge_quality": 0.10,
    },
    "final": {
        "tracking_score": 0.60,
        "reid_score": 0.40,
    },
}
