from pathlib import Path


CASE_ROOT = Path("assets/cases")


DEMO_CASES = [
    {
        "case_id": "case_1_normal_success",
        "title": "Case 1 — Single-Camera",
        "cameras": ["camera_2"],
        "video_files": {
            "camera_2": CASE_ROOT / "case_1_normal_success" / "camera_2.mp4",
        },
        "gt_identity": 3,
        "description": "Pengujian satu kamera.",
    },
    {
        "case_id": "case_2_crowded_fragmentation",
        "title": "Case 2 — Single-Camera",
        "cameras": ["camera_3"],
        "video_files": {
            "camera_3": CASE_ROOT / "case_2_crowded_fragmentation" / "camera_3.mp4",
        },
        "gt_identity": 9,
        "description": "Pengujian satu kamera.",
    },
    {
        "case_id": "case_3_failure_limitation",
        "title": "Case 3 — Single-Camera",
        "cameras": ["camera_1"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_3_failure_limitation" / "camera_1.mp4",
        },
        "gt_identity": 6,
        "description": "Pengujian satu kamera.",
    },
    {
        "case_id": "case_4_multicamera_success",
        "title": "Case 4 — Multi-Camera",
        "cameras": ["camera_1", "camera_3"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_4_multicamera_success" / "camera_1.mp4",
            "camera_3": CASE_ROOT / "case_4_multicamera_success" / "camera_3.mp4",
        },
        "gt_identity": 3,
        "description": "Pengujian multi-kamera.",
    },
    {
        "case_id": "case_5_multicamera_3_video_stress",
        "title": "Case 5 — Multi-Camera",
        "cameras": ["camera_1", "camera_2", "camera_3"],
        "video_files": {
            "camera_1": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_1.mp4",
            "camera_2": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_2.mp4",
            "camera_3": CASE_ROOT / "case_5_multicamera_3_video_stress" / "camera_3.mp4",
        },
        "gt_identity": 9,
        "description": "Pengujian multi-kamera.",
    },
    {
        "case_id": "case_6_multicamera_temporal_handoff",
        "title": "Case 6 — Multi-Camera Temporal",
        "cameras": ["camera_6", "camera_5"],
        "video_files": {
            "camera_6": CASE_ROOT / "case_6_multicamera_temporal_handoff" / "camera_6.mp4",
            "camera_5": CASE_ROOT / "case_6_multicamera_temporal_handoff" / "camera_5.mp4",
        },
        "gt_identity": 3,
        "description": "Pengujian multi-kamera temporal.",
    },
]


def get_case_by_id(case_id):
    for case in DEMO_CASES:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)
