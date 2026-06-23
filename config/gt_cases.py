from pathlib import Path

ANNOTATION_ROOT = Path("assets/annotations")

# Metadata evaluasi GT dipisah dari config utama agar cases.py dan presets.py tetap mengikuti repo lama.
GT_CASE_META = {
    "case_1_normal_success": {
        "annotation_dir": ANNOTATION_ROOT / "case_1",
        "source_frame_start": 1,
        "source_frame_end": 800,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
    "case_2_crowded_fragmentation": {
        "annotation_dir": ANNOTATION_ROOT / "case_2",
        "source_frame_start": 1201,
        "source_frame_end": 2000,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
    "case_3_failure_limitation": {
        "annotation_dir": ANNOTATION_ROOT / "case_3",
        "source_frame_start": 801,
        "source_frame_end": 1600,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
    "case_4_multicamera_success": {
        "annotation_dir": ANNOTATION_ROOT / "case_4",
        "source_frame_start": 1201,
        "source_frame_end": 2000,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
    "case_5_multicamera_3_video_stress": {
        "annotation_dir": ANNOTATION_ROOT / "case_5",
        "source_frame_start": 1201,
        "source_frame_end": 2000,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
    "case_6_multicamera_temporal_handoff": {
        "annotation_dir": ANNOTATION_ROOT / "case_6",
        "source_frame_start": 184,
        "source_frame_end": 1023,
        "local_frame_start": 0,
        "annotation_frame_start": 1,
        "bbox_format": "xyxy",
        "gt_iou_threshold": 0.50,
    },
}


def with_gt_meta(case: dict) -> dict:
    out = dict(case)
    out.update(GT_CASE_META.get(case.get("case_id"), {}))
    return out
