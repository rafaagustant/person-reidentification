from pathlib import Path


CASE_ROOT = Path("assets/cases")


DEMO_CASES = [
    {
        "case_id": "case_1_normal_success",
        "title": "Case 1 - Single-Camera Normal Success",
        "cameras": ["camera_2"],
        "video_files": {
            "camera_2": CASE_ROOT / "case_1_normal_success" / "camera_2.mp4",
        },
        "scene_type": "single_camera_normal",
        "recommended_config": "Recommended config final",
        "gt_identity": 3,
        "description": "Kondisi normal, tiga orang terlihat jelas, overlap rendah.",
        "expected_result": "3 local tracks -> 3 Global ID.",
    },
    {
        "case_id": "case_2_crowded_fragmentation",
        "title": "Case 2 - Crowded / False Positive Sensitive",
        "cameras": ["camera_3"],
        "video_files": {
            "camera_3": CASE_ROOT / "case_2_crowded_fragmentation" / "camera_3.mp4",
        },
        "scene_type": "single_camera_false_positive_sensitive",
        "recommended_config": "Recommended config final",
        "gt_identity": 9,
        "description": "Crowded, occlusion, dan rawan ID switch. Fragment recovery membantu menyambung track utama.",
        "expected_result": "7 local tracks -> 6 Global ID.",
    },
    {
        "case_id": "case_3_failure_limitation",
        "title": "Case 3 - Small/Distant Person Limitation",
        "cameras": ["camera_1"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_3_failure_limitation" / "camera_1.mp4",
        },
        "scene_type": "single_camera_small_or_distant",
        "recommended_config": "Recommended config final",
        "gt_identity": 6,
        "description": "Orang kecil/jauh dari kamera, dipakai sebagai limitation case.",
        "expected_result": "3 local tracks -> 3 Global ID.",
    },
    {
        "case_id": "case_4_multicamera_success",
        "title": "Case 4 - Multi-Camera Same-Time Re-ID",
        "cameras": ["camera_1", "camera_3"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_4_multicamera_success" / "camera_1.mp4",
            "camera_3": CASE_ROOT / "case_4_multicamera_success" / "camera_3.mp4",
        },
        "scene_type": "multi_camera_same_time",
        "recommended_config": "Recommended config final",
        "gt_identity": 3,
        "description": "Dua kamera, cross-camera association + strict intra recovery.",
        "expected_result": "8 local tracks -> 3 Global ID.",
    },
    {
        "case_id": "case_5_multicamera_3_video_stress",
        "title": "Case 5 - Multi-Camera 3-Video Stress Test",
        "cameras": ["camera_1", "camera_2", "camera_3"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_1.mp4",
            "camera_2": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_2.mp4",
            "camera_3": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_3.mp4",
        },
        "scene_type": "multi_camera_crowded_or_stress",
        "recommended_config": "Recommended config final",
        "gt_identity": 9,
        "description": "Tiga kamera, crowded/stress, menggunakan config konservatif.",
        "expected_result": "10 local tracks -> 7 Global ID.",
    },
    {
        "case_id": "case_6_multicamera_temporal_handoff",
        "title": "Case 6 - Multi-Camera Temporal Handoff",
        "cameras": ["camera_6", "camera_5"],
        "video_files": {
            "camera_6": CASE_ROOT / "case_6_multicamera_temporal_handoff" / "camera_6.mp4",
            "camera_5": CASE_ROOT / "case_6_multicamera_temporal_handoff" / "camera_5.mp4",
        },
        "scene_type": "multi_camera_long_term_or_handoff",
        "recommended_config": "Recommended config final",
        "gt_identity": 3,
        "description": "Temporal handoff antar kamera. Enam local track digabung menjadi tiga Global ID.",
        "expected_result": "6 local tracks -> 3 Global ID.",
    },
]


def get_case_by_id(case_id):
    for case in DEMO_CASES:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)
