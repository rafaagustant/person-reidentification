from __future__ import annotations

import copy


VISUAL_PRESETS = {
    "normal_clear": {
        "label": "Normal / jelas",
        "description": "Dipakai jika orang terlihat cukup jelas, ukuran orang sedang-besar, kamera stabil, dan tidak banyak saling menutupi.",
        "tracking": {
            "yolo_conf": 0.10,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.25,
            "match_thresh": 0.80,
            "track_buffer": 30,
        },
        "filter": {
            "min_frames": 50,
            "min_crops": 50,
            "min_avg_conf": 0.45,
            "min_avg_area": 8000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "few_people_clear": {
        "label": "Sedikit orang / jelas",
        "description": "Dipakai jika preview menunjukkan sekitar 1-3 orang, objek terlihat jelas, dan jarang terjadi tumpang tindih antar orang.",
        "tracking": {
            "yolo_conf": 0.20,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.20,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.20,
            "match_thresh": 0.80,
            "track_buffer": 30,
        },
        "filter": {
            "min_frames": 50,
            "min_crops": 50,
            "min_avg_conf": 0.40,
            "min_avg_area": 5000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "crowded_many_people": {
        "label": "Ramai / banyak orang",
        "description": "Dipakai jika preview sering menampilkan minimal 5 orang atau lebih dalam satu frame.",
        "tracking": {
            "yolo_conf": 0.25,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.07,
            "new_track_thresh": 0.25,
            "match_thresh": 0.80,
            "track_buffer": 45,
        },
        "filter": {
            "min_frames": 80,
            "min_crops": 80,
            "min_avg_conf": 0.43,
            "min_avg_area": 3000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "small_distant": {
        "label": "Orang kecil / jauh",
        "description": "Dipakai jika orang terlihat kecil atau jauh dari kamera, misalnya tinggi tubuh kira-kira kurang dari 10-15% tinggi frame.",
        "tracking": {
            "yolo_conf": 0.03,
            "yolo_iou": 0.50,
            "imgsz": 960,
            "track_high_thresh": 0.10,
            "track_low_thresh": 0.03,
            "new_track_thresh": 0.10,
            "match_thresh": 0.88,
            "track_buffer": 8,
        },
        "filter": {
            "min_frames": 15,
            "min_crops": 15,
            "min_avg_conf": 0.28,
            "min_avg_area": 1000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "occlusion_overlap": {
        "label": "Banyak oklusi / saling menutupi",
        "description": "Dipakai jika preview menunjukkan orang sering tertutup objek atau saling menutupi dengan orang lain.",
        "tracking": {
            "yolo_conf": 0.20,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.20,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.20,
            "match_thresh": 0.85,
            "track_buffer": 45,
        },
        "filter": {
            "min_frames": 40,
            "min_crops": 40,
            "min_avg_conf": 0.35,
            "min_avg_area": 2500,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "low_quality_blur": {
        "label": "Video blur / gelap",
        "description": "Dipakai jika preview terlihat blur, gelap, atau kualitas visual kurang jelas.",
        "tracking": {
            "yolo_conf": 0.10,
            "yolo_iou": 0.50,
            "imgsz": 960,
            "track_high_thresh": 0.15,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.15,
            "match_thresh": 0.80,
            "track_buffer": 30,
        },
        "filter": {
            "min_frames": 30,
            "min_crops": 30,
            "min_avg_conf": 0.30,
            "min_avg_area": 1500,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
}


TRACKING_PRESETS = {
    "tracking_balanced": {
        "label": "Balanced",
        "description": "Default untuk kondisi normal.",
        "tracking": {
            "yolo_conf": 0.10,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.25,
            "match_thresh": 0.80,
            "track_buffer": 30,
        },
    },
    "tracking_precision": {
        "label": "Precision",
        "description": "Lebih ketat untuk mengurangi false positive.",
        "tracking": {
            "yolo_conf": 0.30,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.30,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.30,
            "match_thresh": 0.80,
            "track_buffer": 30,
        },
    },
    "tracking_case_2_final": {
        "label": "Case 2 final",
        "description": "Konfigurasi final tracking untuk case 2 crowded/fragmentation.",
        "tracking": {
            "yolo_conf": 0.20,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.20,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.20,
            "match_thresh": 0.80,
            "track_buffer": 45,
        },
    },
    "tracking_recall": {
        "label": "Recall",
        "description": "Lebih longgar untuk objek kecil atau jauh.",
        "tracking": {
            "yolo_conf": 0.05,
            "yolo_iou": 0.50,
            "imgsz": 1280,
            "track_high_thresh": 0.10,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.10,
            "match_thresh": 0.75,
            "track_buffer": 30,
        },
    },
    "tracking_case_3_final": {
        "label": "Case 3 final",
        "description": "Konfigurasi final untuk case 3 orang kecil/jauh.",
        "tracking": {
            "yolo_conf": 0.03,
            "yolo_iou": 0.50,
            "imgsz": 960,
            "track_high_thresh": 0.10,
            "track_low_thresh": 0.03,
            "new_track_thresh": 0.10,
            "match_thresh": 0.88,
            "track_buffer": 8,
        },
    },
    "tracking_multicam_balanced": {
        "label": "Multi-camera balanced",
        "description": "Tracking seimbang untuk case lintas kamera.",
        "tracking": {
            "yolo_conf": 0.30,
            "yolo_iou": 0.50,
            "imgsz": 640,
            "track_high_thresh": 0.30,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.30,
            "match_thresh": 0.80,
            "track_buffer": 30

        },
    },
}


FILTER_PRESETS = {
    "filter_normal": {
        "label": "Normal",
        "description": "Filter standar untuk track yang jelas.",
        "filter": {
            "min_frames": 50,
            "min_crops": 50,
            "min_avg_conf": 0.45,
            "min_avg_area": 15000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_crowded": {
        "label": "Crowded",
        "description": "Lebih ketat untuk menekan track palsu.",
        "filter": {
            "min_frames": 80,
            "min_crops": 80,
            "min_avg_conf": 0.45,
            "min_avg_area": 8000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_case_2_final": {
        "label": "Case 2 final",
        "description": "Filter final untuk case 2 crowded/fragmentation.",
        "filter": {
            "min_frames": 80,
            "min_crops": 80,
            "min_avg_conf": 0.40,
            "min_avg_area": 3000.0,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_small_person": {
        "label": "Small person",
        "description": "Lebih longgar untuk orang kecil atau jauh.",
        "filter": {
            "min_frames": 10,
            "min_crops": 10,
            "min_avg_conf": 0.25,
            "min_avg_area": 800,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_case_3_final": {
        "label": "Case 3 final",
        "description": "Filter final untuk mempertahankan track kecil/jauh pada case 3.",
        "filter": {
            "min_frames": 15,
            "min_crops": 15,
            "min_avg_conf": 0.28,
            "min_avg_area": 1000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_multicam": {
        "label": "Multi-camera",
        "description": "Filter longgar agar kandidat Re-ID lintas kamera tidak terlalu cepat dibuang.",
        "filter": {
            "min_frames": 25,
            "min_crops": 25,
            "min_avg_conf": 0.30,
            "min_avg_area": 2000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_multicam_strict": {
        "label": "Multi-camera strict",
        "description": "Filter lebih ketat untuk stress test.",
        "filter": {
            "min_frames": 80,
            "min_crops": 80,
            "min_avg_conf": 0.45,
            "min_avg_area": 8000,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
    "filter_temporal_handoff": {
        "label": "Temporal handoff",
        "description": "Menjaga track yang cukup pendek tetapi masih layak untuk Re-ID.",
        "filter": {
            "min_frames": 20,
            "min_crops": 20,
            "min_avg_conf": 0.30,
            "min_avg_area": 1500,
            "max_samples_per_track": 32,
            "crop_selection_strategy": "quality",
        },
    },
}


REID_PRESETS = {
    "reid_single_camera": {
        "label": "Single-camera",
        "description": "Tidak melakukan merge antar kamera.",
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": False,
            "cross_threshold": 0.75,
            "intra_threshold": 0.80,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_case_3_final": {
        "label": "Case 3 final",
        "description": "Strict intra-camera recovery final untuk case 3 tanpa cross-camera merge.",
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.66,
            "intra_max_gap": 220,
            "intra_max_overlap": 15,
            "use_mnn": True,
        },
    },
    "reid_intra_recovery": {
        "label": "Fragment recovery",
        "description": "Menyambung track terputus pada kamera yang sama.",
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.75,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_case_2_final": {
        "label": "Case 2 final",
        "description": "Konfigurasi final Re-ID untuk case 2 fragment recovery.",
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.78,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_cross_camera_balanced": {
        "label": "Cross-camera balanced",
        "description": "Merge lintas kamera dengan threshold sedang dan MNN.",
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.75,
            "intra_max_gap": 250,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_case_4_final": {
        "label": "Case 4 final",
        "description": "Konfigurasi final Re-ID untuk case 4 multi-camera success.",
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.74,
            "intra_max_gap": 450,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_cross_camera_conservative": {
        "label": "Cross-camera conservative",
        "description": "Lebih ketat untuk mengurangi false merge.",
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.79,
            "intra_threshold": 0.80,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_case_5_final": {
        "label": "Case 5 final",
        "description": "Konfigurasi final Re-ID untuk case 5 multi-camera stress.",
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.78,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
    "reid_temporal_handoff": {
        "label": "Temporal handoff",
        "description": "Asosiasi lintas kamera untuk perpindahan antar waktu.",
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": False,
            "cross_threshold": 0.79,
            "intra_threshold": 0.80,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
    },
}

CASE_CAMERA_CONFIG_RECOMMENDATIONS = {
    "case_4_multicamera_success": {
        "camera_1": {
            "profile": "manual_case_4_camera_1",
            "tracking": {
                "yolo_conf": 0.20,
                "yolo_iou": 0.50,
                "imgsz": 640,
                "track_high_thresh": 0.20,
                "track_low_thresh": 0.05,
                "new_track_thresh": 0.20,
                "match_thresh": 0.80,
                "track_buffer": 30,
            },
            "filter": {
                "min_frames": 60,
                "min_crops": 60,
                "min_avg_conf": 0.30,
                "min_avg_area": 2500,
                "max_samples_per_track": 32,
                "crop_selection_strategy": "quality",
            },
        },
        "camera_3": {
            "profile": "manual_case_4_camera_3",
            "tracking": {
                "yolo_conf": 0.20,
                "yolo_iou": 0.50,
                "imgsz": 640,
                "track_high_thresh": 0.20,
                "track_low_thresh": 0.05,
                "new_track_thresh": 0.20,
                "match_thresh": 0.80,
                "track_buffer": 30,
            },
            "filter": {
                "min_frames": 60,
                "min_crops": 60,
                "min_avg_conf": 0.30,
                "min_avg_area": 2500,
                "max_samples_per_track": 32,
                "crop_selection_strategy": "quality",
            },
        },
    },
    "case_6_multicamera_temporal_handoff": {
        "camera_6": {
            "profile": "manual_case_6_camera_6",
            "tracking": {
                "yolo_conf": 0.30,
                "yolo_iou": 0.50,
                "imgsz": 640,
                "track_high_thresh": 0.30,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.30,
                "match_thresh": 0.80,
                "track_buffer": 30,
            },
            "filter": {
                "min_frames": 80,
                "min_crops": 80,
                "min_avg_conf": 0.45,
                "min_avg_area": 8000,
                "max_samples_per_track": 32,
                "crop_selection_strategy": "quality",
            },
        },
        "camera_5": {
            "profile": "manual_case_6_camera_5",
            "tracking": {
                "yolo_conf": 0.30,
                "yolo_iou": 0.50,
                "imgsz": 640,
                "track_high_thresh": 0.30,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.30,
                "match_thresh": 0.80,
                "track_buffer": 30,
            },
            "filter": {
                "min_frames": 80,
                "min_crops": 80,
                "min_avg_conf": 0.45,
                "min_avg_area": 8000,
                "max_samples_per_track": 32,
                "crop_selection_strategy": "quality",
            },
        },
    },
}


CASE_RECOMMENDATIONS = {
    "case_1_normal_success": {
        "camera_configs": {
            "camera_2": {
                "profile": "case_1_final_camera_2",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.45,
                    "min_avg_area": 8000,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": False,
            "cross_threshold": 0.75,
            "intra_threshold": 0.80,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "note": "Config final Case 1: single-camera normal, precision tinggi, tanpa fragment recovery.",
    },
    "case_2_crowded_fragmentation": {
        "camera_configs": {
            "camera_3": {
                "profile": "case_2_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.20,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.20,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.20,
                    "match_thresh": 0.80,
                    "track_buffer": 45,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.40,
                    "min_avg_area": 3000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.78,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "note": "Config final Case 2: crowded/occlusion, filter cukup ketat, intra-camera fragment recovery aktif.",
    },
    "case_3_failure_limitation": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_3_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.03,
                    "yolo_iou": 0.60,
                    "imgsz": 1280,
                    "track_high_thresh": 0.10,
                    "track_low_thresh": 0.03,
                    "new_track_thresh": 0.10,
                    "match_thresh": 0.84,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 15,
                    "min_crops": 15,
                    "min_avg_conf": 0.25,
                    "min_avg_area": 500.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": False,
            "enable_strict_intra": True,
            "cross_threshold": 0.75,
            "intra_threshold": 0.66,
            "intra_max_gap": 220,
            "intra_max_overlap": 15,
            "use_mnn": True,
        },
        "note": "Config final Case 3: limitation case untuk orang kecil/jauh. Hasil tetap dicatat sebagai keterbatasan sistem.",
    },
    "case_4_multicamera_success": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_4_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.20,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.20,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.20,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 60,
                    "min_crops": 60,
                    "min_avg_conf": 0.30,
                    "min_avg_area": 2500.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_3": {
                "profile": "case_4_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.15,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.15,
                    "track_low_thresh": 0.04,
                    "new_track_thresh": 0.15,
                    "match_thresh": 0.88,
                    "track_buffer": 90,
                },
                "filter": {
                    "min_frames": 30,
                    "min_crops": 30,
                    "min_avg_conf": 0.27,
                    "min_avg_area": 2000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.74,
            "intra_max_gap": 450,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "note": "Config final Case 4: multi-camera success, 6 local tracks menjadi 3 Global ID.",
    },
    "case_5_multicamera_3_video_stress": {
        "camera_configs": {
            "camera_1": {
                "profile": "case_5_final_camera_1",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 10,
                    "min_crops": 10,
                    "min_avg_conf": 0.35,
                    "min_avg_area": 3000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_2": {
                "profile": "case_5_final_camera_2",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.45,
                    "min_avg_area": 8000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_3": {
                "profile": "case_5_final_camera_3",
                "tracking": {
                    "yolo_conf": 0.30,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.30,
                    "track_low_thresh": 0.10,
                    "new_track_thresh": 0.30,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 80,
                    "min_crops": 80,
                    "min_avg_conf": 0.43,
                    "min_avg_area": 8000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": True,
            "cross_threshold": 0.74,
            "intra_threshold": 0.78,
            "intra_max_gap": 450,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "note": "Config final Case 5: stress test 3 kamera, menjaga purity dan meningkatkan fragment recovery.",
    },
    "case_6_multicamera_temporal_handoff": {
        "camera_configs": {
            "camera_6": {
                "profile": "case_6_final_camera_6",
                "tracking": {
                    "yolo_conf": 0.25,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.07,
                    "new_track_thresh": 0.25,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 45,
                    "min_crops": 45,
                    "min_avg_conf": 0.55,
                    "min_avg_area": 15000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
            "camera_5": {
                "profile": "case_6_final_camera_5",
                "tracking": {
                    "yolo_conf": 0.25,
                    "yolo_iou": 0.50,
                    "imgsz": 640,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.07,
                    "new_track_thresh": 0.25,
                    "match_thresh": 0.80,
                    "track_buffer": 30,
                },
                "filter": {
                    "min_frames": 45,
                    "min_crops": 45,
                    "min_avg_conf": 0.55,
                    "min_avg_area": 15000.0,
                    "max_samples_per_track": 32,
                    "crop_selection_strategy": "quality",
                },
            },
        },
        "reid": {
            "enable_cross_camera": True,
            "enable_strict_intra": False,
            "cross_threshold": 0.80,
            "intra_threshold": 0.00,
            "intra_max_gap": 30,
            "intra_max_overlap": 0,
            "use_mnn": True,
        },
        "note": "Config final Case 6: temporal handoff, 6 local tracks menjadi 3 Global ID dengan score 1.0.",
    },
}


TRACKING_CONFIGS = {
    "Tracking-Balanced": {
        **TRACKING_PRESETS["tracking_balanced"]["tracking"],
        "description": TRACKING_PRESETS["tracking_balanced"]["description"],
    },
    "Tracking-Precision": {
        **TRACKING_PRESETS["tracking_precision"]["tracking"],
        "description": TRACKING_PRESETS["tracking_precision"]["description"],
    },
    "Tracking-Recall": {
        **TRACKING_PRESETS["tracking_recall"]["tracking"],
        "description": TRACKING_PRESETS["tracking_recall"]["description"],
    },
}


CONFIG_PRESETS = {
    "Balanced Default": {
        "tracking_preset": "tracking_balanced",
        "filter_preset": "filter_normal",
        "reid_preset": "reid_single_camera",
        "tracking_config": "Tracking-Balanced",
        "description": "Default single-camera normal.",
    },
    "High Precision": {
        "tracking_preset": "tracking_precision",
        "filter_preset": "filter_crowded",
        "reid_preset": "reid_intra_recovery",
        "tracking_config": "Tracking-Precision",
        "description": "Crowded dan fragment recovery.",
    },
    "High Recall": {
        "tracking_preset": "tracking_recall",
        "filter_preset": "filter_small_person",
        "reid_preset": "reid_single_camera",
        "tracking_config": "Tracking-Recall",
        "description": "Fallback untuk orang kecil/jauh.",
    },
    "Multi-Camera Success": {
        "tracking_preset": "tracking_multicam_balanced",
        "filter_preset": "filter_multicam",
        "reid_preset": "reid_cross_camera_balanced",
        "tracking_config": "Tracking-Balanced",
        "description": "Multi-camera same-time atau handoff.",
    },
    "Multi-Camera Conservative": {
        "tracking_preset": "tracking_precision",
        "filter_preset": "filter_multicam_strict",
        "reid_preset": "reid_cross_camera_conservative",
        "tracking_config": "Tracking-Precision",
        "description": "Multi-camera crowded/stress.",
    },
}


FINAL_CASE_POLICY_OVERRIDES = {}


LEGACY_PRESET_TO_STAGE_KEYS = {
    name: (
        cfg["tracking_preset"],
        cfg["filter_preset"],
        cfg["reid_preset"],
    )
    for name, cfg in CONFIG_PRESETS.items()
}


def get_case_recommendation(case_id: str) -> dict:
    if case_id not in CASE_RECOMMENDATIONS:
        raise KeyError(case_id)
    return copy.deepcopy(CASE_RECOMMENDATIONS[case_id])


def get_case_camera_configs(case_id: str) -> dict:
    rec = CASE_RECOMMENDATIONS.get(case_id, {})
    return copy.deepcopy(rec.get("camera_configs", {}))


def get_default_stage_keys(case_id: str) -> tuple[str, str, str]:
    return "tracking_balanced", "filter_normal", "reid_single_camera"


def build_config_from_case_recommendation(case_id: str) -> dict:
    rec = get_case_recommendation(case_id)
    camera_configs = copy.deepcopy(rec.get("camera_configs", {}))
    first_camera_cfg = next(iter(camera_configs.values()), {})
    tracking = copy.deepcopy(first_camera_cfg.get("tracking") or TRACKING_PRESETS["tracking_balanced"]["tracking"])
    filter_cfg = copy.deepcopy(first_camera_cfg.get("filter") or FILTER_PRESETS["filter_normal"]["filter"])
    return {
        "tracking_config_mode": "per_camera",
        "tracking": tracking,
        "filter": filter_cfg,
        "reid": copy.deepcopy(rec.get("reid") or REID_PRESETS["reid_single_camera"]["reid"]),
        "camera_configs": camera_configs,
        "recommended_note": rec.get("note", ""),
    }


def build_config_from_stage_presets(
    tracking_key: str,
    filter_key: str,
    reid_key: str,
) -> dict:
    if tracking_key not in TRACKING_PRESETS:
        raise KeyError(tracking_key)
    if filter_key not in FILTER_PRESETS:
        raise KeyError(filter_key)
    if reid_key not in REID_PRESETS:
        raise KeyError(reid_key)

    return {
        "tracking_config_mode": "per_camera",
        "tracking_preset": tracking_key,
        "filter_preset": filter_key,
        "reid_preset": reid_key,
        "tracking": copy.deepcopy(TRACKING_PRESETS[tracking_key]["tracking"]),
        "filter": copy.deepcopy(FILTER_PRESETS[filter_key]["filter"]),
        "reid": copy.deepcopy(REID_PRESETS[reid_key]["reid"]),
        "camera_configs": {},
    }


def _with_defaults(section: dict, defaults: dict, casts: dict) -> dict:
    out = copy.deepcopy(defaults)
    out.update(section or {})
    for key, caster in casts.items():
        out[key] = caster(out[key])
    return out


def normalize_config(config: dict | None) -> dict:
    config = copy.deepcopy(config or {})

    tracking_key = config.get("tracking_preset")
    filter_key = config.get("filter_preset")
    reid_key = config.get("reid_preset")

    legacy_preset = config.get("preset") or config.get("recommended_config")
    if legacy_preset in LEGACY_PRESET_TO_STAGE_KEYS:
        tracking_key, filter_key, reid_key = LEGACY_PRESET_TO_STAGE_KEYS[legacy_preset]

    if not tracking_key:
        legacy_tracking = config.get("tracking_config", "Tracking-Balanced")
        tracking_key = {
            "Tracking-Precision": "tracking_precision",
            "Tracking-Recall": "tracking_recall",
        }.get(legacy_tracking, "tracking_balanced")
    if not filter_key:
        filter_key = "filter_normal"
    if not reid_key:
        reid_key = "reid_single_camera"

    base = build_config_from_stage_presets(tracking_key, filter_key, reid_key)

    tracking_flat_keys = set(base["tracking"])
    filter_flat_keys = set(base["filter"])
    reid_flat_keys = set(base["reid"])

    tracking_updates = {k: config[k] for k in tracking_flat_keys if k in config}
    filter_updates = {k: config[k] for k in filter_flat_keys if k in config}
    reid_updates = {k: config[k] for k in reid_flat_keys if k in config}

    base["tracking"].update(config.get("tracking", {}))
    base["filter"].update(config.get("filter", {}))
    base["reid"].update(config.get("reid", {}))
    base["tracking"].update(tracking_updates)
    base["filter"].update(filter_updates)
    base["reid"].update(reid_updates)

    if "min_frames" not in base["filter"] and "min_crops" in base["filter"]:
        base["filter"]["min_frames"] = base["filter"]["min_crops"]
    if "min_crops" not in base["filter"] and "min_frames" in base["filter"]:
        base["filter"]["min_crops"] = base["filter"]["min_frames"]

    tracking_defaults = TRACKING_PRESETS[tracking_key]["tracking"]
    filter_defaults = FILTER_PRESETS[filter_key]["filter"]
    reid_defaults = REID_PRESETS[reid_key]["reid"]

    base["tracking"] = _with_defaults(
        base["tracking"],
        tracking_defaults,
        {
            "yolo_conf": float,
            "yolo_iou": float,
            "imgsz": int,
            "track_high_thresh": float,
            "track_low_thresh": float,
            "new_track_thresh": float,
            "match_thresh": float,
            "track_buffer": int,
        },
    )
    base["filter"] = _with_defaults(
        base["filter"],
        filter_defaults,
        {
            "min_frames": int,
            "min_crops": int,
            "min_avg_conf": float,
            "min_avg_area": float,
            "max_samples_per_track": int,
            "crop_selection_strategy": str,
        },
    )
    base["reid"] = _with_defaults(
        base["reid"],
        reid_defaults,
        {
            "enable_cross_camera": bool,
            "enable_strict_intra": bool,
            "cross_threshold": float,
            "intra_threshold": float,
            "intra_max_gap": int,
            "intra_max_overlap": int,
            "use_mnn": bool,
        },
    )

    base["custom"] = copy.deepcopy(config.get("custom", {}))
    if "visual_preset" in config:
        base["visual_preset"] = copy.deepcopy(config["visual_preset"])
    base["tracking_config_mode"] = "per_camera"
    base["camera_configs"] = copy.deepcopy(config.get("camera_configs", {}))
    if "raw" in config:
        base["raw"] = copy.deepcopy(config["raw"])
    return base


def get_camera_stage_config(config: dict | None, camera_name: str) -> tuple[dict, dict]:
    """
    Return tracking_cfg dan filter_cfg untuk kamera tertentu.
    Fallback ke config global jika camera-specific config tidak tersedia.
    """
    cfg = normalize_config(config)
    tracking_cfg = copy.deepcopy(cfg.get("tracking", {}))
    filter_cfg = copy.deepcopy(cfg.get("filter", {}))
    camera_cfg = (cfg.get("camera_configs") or {}).get(camera_name, {}) or {}

    tracking_cfg.update(copy.deepcopy(camera_cfg.get("tracking") or {}))
    filter_cfg.update(copy.deepcopy(camera_cfg.get("filter") or {}))
    tracking_cfg = _with_defaults(
        tracking_cfg,
        cfg["tracking"],
        {
            "yolo_conf": float,
            "yolo_iou": float,
            "imgsz": int,
            "track_high_thresh": float,
            "track_low_thresh": float,
            "new_track_thresh": float,
            "match_thresh": float,
            "track_buffer": int,
        },
    )
    filter_cfg = _with_defaults(
        filter_cfg,
        cfg["filter"],
        {
            "min_frames": int,
            "min_crops": int,
            "min_avg_conf": float,
            "min_avg_area": float,
            "max_samples_per_track": int,
            "crop_selection_strategy": str,
        },
    )
    return tracking_cfg, filter_cfg
