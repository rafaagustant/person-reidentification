from __future__ import annotations

import copy

import streamlit as st

from config.presets import (
    build_config_from_case_recommendation,
    build_error_analysis_config,
    normalize_config,
)


def configuration_for_mode(case_id: str, mode: str) -> dict:
    config = (
        build_config_from_case_recommendation(case_id)
        if mode == "Rekomendasi Case"
        else build_error_analysis_config(case_id)
    )
    config["config_mode"] = "case_recommendation" if mode == "Rekomendasi Case" else "error_analysis_tuning"
    return normalize_config(config)


def _number_inputs(config: dict) -> dict:
    edited = copy.deepcopy(config)
    tracking, filtering, reid = edited["tracking"], edited["filter"], edited["reid"]
    with st.expander("Deteksi dan tracking", expanded=False):
        tracking["yolo_conf"] = st.number_input("YOLO confidence", 0.0, 1.0, float(tracking["yolo_conf"]), 0.01)
        tracking["yolo_iou"] = st.number_input("YOLO IoU", 0.0, 1.0, float(tracking["yolo_iou"]), 0.01)
        tracking["imgsz"] = st.number_input("Ukuran citra", 32, 1920, int(tracking["imgsz"]), 32)
        tracking["track_high_thresh"] = st.number_input("Track high threshold", 0.0, 1.0, float(tracking["track_high_thresh"]), 0.01)
        tracking["track_low_thresh"] = st.number_input("Track low threshold", 0.0, 1.0, float(tracking["track_low_thresh"]), 0.01)
        tracking["new_track_thresh"] = st.number_input("New track threshold", 0.0, 1.0, float(tracking["new_track_thresh"]), 0.01)
        tracking["match_thresh"] = st.number_input("BoT-SORT match threshold", 0.0, 1.0, float(tracking["match_thresh"]), 0.01)
        tracking["track_buffer"] = st.number_input("Track buffer", 0, 600, int(tracking["track_buffer"]), 1)
    with st.expander("Penyaringan track", expanded=False):
        filtering["min_frames"] = st.number_input("Minimum frame", 0, 10000, int(filtering["min_frames"]), 1)
        filtering["min_crops"] = st.number_input("Minimum crop", 0, 10000, int(filtering["min_crops"]), 1)
        filtering["min_avg_conf"] = st.number_input("Rata-rata confidence minimum", 0.0, 1.0, float(filtering["min_avg_conf"]), 0.01)
        filtering["min_avg_area"] = st.number_input("Rata-rata area minimum", 0.0, 10000000.0, float(filtering["min_avg_area"]), 100.0)
        filtering["max_samples_per_track"] = st.number_input("Maksimum sampel per track", 1, 256, int(filtering["max_samples_per_track"]), 1)
        filtering["crop_selection_strategy"] = st.selectbox("Strategi sampel crop", ["quality", "uniform"], index=["quality", "uniform"].index(filtering["crop_selection_strategy"]))
    with st.expander("Asosiasi Re-ID", expanded=False):
        reid["enable_cross_camera"] = st.checkbox("Aktifkan lintas kamera", bool(reid["enable_cross_camera"]))
        reid["enable_strict_intra"] = st.checkbox("Aktifkan pemulihan fragmentasi intra-kamera", bool(reid["enable_strict_intra"]))
        reid["use_mnn"] = st.checkbox("Gunakan mutual nearest neighbour", bool(reid["use_mnn"]))
        reid["cross_threshold"] = st.number_input("Cross-camera threshold", 0.0, 1.0, float(reid["cross_threshold"]), 0.01)
        reid["intra_threshold"] = st.number_input("Intra-camera threshold", 0.0, 1.0, float(reid["intra_threshold"]), 0.01)
        reid["intra_max_gap"] = st.number_input("Maximum temporal gap", 0, 100000, int(reid["intra_max_gap"]), 1)
        reid["intra_max_overlap"] = st.number_input("Maximum temporal overlap", 0, 100000, int(reid["intra_max_overlap"]), 1)
    return edited


def render_config_panel(case: dict, state: dict) -> dict:
    mode = st.segmented_control(
        "Mode konfigurasi", ["Rekomendasi Case", "Tuning Analisis Kesalahan"],
        default=state.get("configuration_mode", "Rekomendasi Case"), key=f"mode_{case['case_id']}",
    ) or "Rekomendasi Case"
    if state.get("configuration_mode") != mode or state.get("config") is None:
        state["configuration_mode"] = mode
        state["config"] = configuration_for_mode(case["case_id"], mode)
    config = copy.deepcopy(state["config"])
    st.caption(config.get("recommended_note", "Konfigurasi awal permisif untuk mengidentifikasi kesalahan dominan."))
    st.json({"tracking": config["tracking"], "filter": config["filter"], "reid": config["reid"]}, expanded=False)
    with st.form(f"config_form_{case['case_id']}", border=True):
        edited = _number_inputs(config)
        submitted = st.form_submit_button("Terapkan parameter", icon=":material/tune:")
    if submitted:
        edited["config_mode"] = config["config_mode"]
        # Global edits are applied to each camera unless the researcher later restores a case recommendation.
        for camera in edited.get("camera_configs", {}):
            edited["camera_configs"][camera]["tracking"] = copy.deepcopy(edited["tracking"])
            edited["camera_configs"][camera]["filter"] = copy.deepcopy(edited["filter"])
        state["config"] = normalize_config(edited)
        st.success("Parameter diterapkan untuk run berikutnya.")
    return state["config"]
