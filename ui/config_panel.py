from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from config.presets import build_config_from_case_recommendation, build_error_analysis_config, normalize_config
from ui.components import widget_key
from ui.state import invalidate_for_config_change


PARAMETER_HELP = {
    "yolo_conf": "Ambang minimum confidence deteksi manusia.", "yolo_iou": "Ambang IoU untuk proses Non-Maximum Suppression pada deteksi YOLO11n.",
    "imgsz": "Ukuran citra masukan yang digunakan YOLO11n.", "track_high_thresh": "Ambang deteksi berkepercayaan tinggi pada asosiasi pertama BoT-SORT.",
    "track_low_thresh": "Ambang deteksi berkepercayaan rendah pada asosiasi kedua BoT-SORT.", "new_track_thresh": "Ambang untuk membentuk local track baru.",
    "match_thresh": "Ambang pencocokan deteksi dengan track pada BoT-SORT.", "track_buffer": "Jumlah frame untuk mempertahankan track yang hilang sementara.",
    "min_frames": "Jumlah minimum frame agar local track dinyatakan valid.", "min_crops": "Jumlah minimum crop agar local track dinyatakan valid.",
    "min_avg_conf": "Rata-rata confidence minimum pada local track.", "min_avg_area": "Rata-rata luas bounding box minimum pada local track.",
    "max_samples_per_track": "Jumlah maksimum crop yang digunakan untuk membentuk embedding track.", "crop_selection_strategy": "Strategi pemilihan crop representatif dari setiap local track.",
    "enable_cross_camera": "Mengaktifkan asosiasi lintas kamera.", "enable_strict_intra": "Mengaktifkan pemulihan fragmentasi dalam kamera.", "use_mnn": "Mengaktifkan mutual nearest neighbour pada asosiasi lintas kamera.",
    "cross_threshold": "Ambang cosine similarity untuk asosiasi lintas kamera.", "intra_threshold": "Ambang cosine similarity untuk pemulihan fragmentasi dalam kamera.",
    "intra_max_gap": "Jeda frame maksimum untuk asosiasi intra-camera.", "intra_max_overlap": "Tumpang tindih frame maksimum yang masih diperbolehkan.",
}

TRACKING_KEYS = ["yolo_conf", "yolo_iou", "imgsz", "track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh", "track_buffer"]
FILTER_KEYS = ["min_frames", "min_crops", "min_avg_conf", "min_avg_area", "max_samples_per_track", "crop_selection_strategy"]
REID_KEYS = ["enable_cross_camera", "enable_strict_intra", "use_mnn", "cross_threshold", "intra_threshold", "intra_max_gap", "intra_max_overlap"]
ALLOWED_TUNING_IMGSZ = [640, 960, 1280]


def configuration_for_mode(case_id: str, mode: str) -> dict:
    config = build_config_from_case_recommendation(case_id) if mode == "Konfigurasi Rekomendasi" else build_error_analysis_config(case_id)
    config["config_mode"] = "case_recommendation" if mode == "Konfigurasi Rekomendasi" else "initial_analysis"
    return normalize_config(config)


def _parameter_table(values: dict, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"parameter": name, "nilai": str(values.get(name)), "keterangan": PARAMETER_HELP[name]} for name in keys])


def _recommended_view(case_id: str, config: dict) -> None:
    st.caption("Konfigurasi rekomendasi merupakan konfigurasi terpilih berdasarkan hasil evaluasi dan penyesuaian parameter pada setiap case.")
    cameras = list(config.get("camera_configs", {}))
    camera_tabs = st.tabs(cameras) if cameras else []
    for tab, camera in zip(camera_tabs, cameras):
        with tab:
            camera_config = config["camera_configs"][camera]
            left, right = st.columns(2)
            with left:
                st.markdown("**Deteksi dan Tracking**")
                st.dataframe(_parameter_table(camera_config.get("tracking", {}), TRACKING_KEYS), hide_index=True, key=widget_key(case_id, "recommendation", "tracking_table", camera))
            with right:
                st.markdown("**Penyaringan Track**")
                st.dataframe(_parameter_table(camera_config.get("filter", {}), FILTER_KEYS), hide_index=True, key=widget_key(case_id, "recommendation", "filter_table", camera))
    st.markdown("**Asosiasi Re-ID**")
    st.dataframe(_parameter_table(config.get("reid", {}), REID_KEYS), hide_index=True, key=widget_key(case_id, "recommendation", "reid_table"))


def _number(
    config: dict,
    name: str,
    case_id: str,
    section: str,
    camera: str | None,
    integer: bool = False,
    minimum: float = 0,
    maximum: float | int = 1,
) -> None:
    value = config[name]

    if integer:
        min_value = int(minimum)
        max_value = int(maximum)
        input_value = int(value)
        step = 1
    else:
        min_value = float(minimum)
        max_value = float(maximum)
        input_value = float(value)
        step = 0.01

    config[name] = st.number_input(
        f"`{name}`",
        min_value=min_value,
        max_value=max_value,
        value=input_value,
        step=step,
        help=PARAMETER_HELP[name],
        key=widget_key(case_id, section, name, camera),
    )


def _controls(case_id: str, camera: str | None, tracking: dict, filtering: dict, reid: dict) -> tuple[dict, dict, dict]:
    tracking, filtering, reid = copy.deepcopy(tracking), copy.deepcopy(filtering), copy.deepcopy(reid)
    with st.expander("Deteksi dan Tracking", expanded=True):
        detection, botsort = st.columns(2)
        with detection:
            _number(tracking, "yolo_conf", case_id, "analysis_detection", camera)
            _number(tracking, "yolo_iou", case_id, "analysis_detection", camera, minimum=0.01)
            tracking["imgsz"] = int(st.selectbox(
                "`imgsz`",
                ALLOWED_TUNING_IMGSZ,
                index=ALLOWED_TUNING_IMGSZ.index(int(tracking["imgsz"])) if int(tracking["imgsz"]) in ALLOWED_TUNING_IMGSZ else ALLOWED_TUNING_IMGSZ.index(960),
                help="Ukuran citra masukan YOLO11n. Pilihan pengujian dibatasi pada 640, 960, dan 1280.",
                key=widget_key(case_id, "analysis_detection", "imgsz", camera),
            ))
        with botsort:
            for name in ["track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"]:
                _number(tracking, name, case_id, "analysis_tracking", camera)
            _number(tracking, "track_buffer", case_id, "analysis_tracking", camera, integer=True, minimum=0, maximum=1000)
    with st.expander("Penyaringan Track"):
        first, second = st.columns(2)
        with first:
            for name in ["min_frames", "min_crops"]:
                _number(filtering, name, case_id, "analysis_filter", camera, integer=True, minimum=0, maximum=100000)
            _number(filtering, "min_avg_conf", case_id, "analysis_filter", camera)
        with second:
            _number(filtering, "min_avg_area", case_id, "analysis_filter", camera, minimum=0, maximum=10000000)
            _number(filtering, "max_samples_per_track", case_id, "analysis_filter", camera, integer=True, minimum=1, maximum=256)
            filtering["crop_selection_strategy"] = st.selectbox("`crop_selection_strategy`", ["quality", "uniform"], index=["quality", "uniform"].index(filtering["crop_selection_strategy"]), help=PARAMETER_HELP["crop_selection_strategy"], key=widget_key(case_id, "analysis_filter", "crop_selection_strategy", camera))
    with st.expander("Asosiasi Re-ID"):
        left, right = st.columns(2)
        with left:
            for name in ["enable_cross_camera", "enable_strict_intra", "use_mnn"]:
                reid[name] = st.checkbox(f"`{name}`", value=bool(reid[name]), help=PARAMETER_HELP[name], key=widget_key(case_id, "analysis_reid", name))
            _number(reid, "cross_threshold", case_id, "analysis_reid", None)
            _number(reid, "intra_threshold", case_id, "analysis_reid", None)
        with right:
            _number(reid, "intra_max_gap", case_id, "analysis_reid", None, integer=True, minimum=0, maximum=100000)
            _number(reid, "intra_max_overlap", case_id, "analysis_reid", None, integer=True, minimum=0, maximum=100000)
    return tracking, filtering, reid


def _apply_scope(config: dict, tracking: dict, filtering: dict, reid: dict, scope: str, camera: str | None) -> dict:
    updated = copy.deepcopy(config)
    targets = list(updated.get("camera_configs", {})) if scope == "Semua kamera" else [camera]
    for target in targets:
        if target:
            updated["camera_configs"][target]["tracking"] = copy.deepcopy(tracking)
            updated["camera_configs"][target]["filter"] = copy.deepcopy(filtering)
    updated["reid"] = copy.deepcopy(reid)
    return updated


def render_config_panel(case: dict, state: dict) -> dict:
    case_id = case["case_id"]
    modes = ["Konfigurasi Rekomendasi", "Konfigurasi Awal Analisis"]
    default = state.get("configuration_mode", modes[0])
    mode = st.segmented_control("Sumber konfigurasi", modes, default=default if default in modes else modes[0], key=widget_key(case_id, "configuration", "mode")) or modes[0]
    if state.get("config") is None or state.get("configuration_mode") != mode:
        message = invalidate_for_config_change(state, state.get("config"), configuration_for_mode(case_id, mode), case)
        state["configuration_mode"] = mode
        if message:
            st.info(message)
    config = copy.deepcopy(state["config"])
    if mode == "Konfigurasi Rekomendasi":
        _recommended_view(case_id, config)
    else:
        st.caption("Konfigurasi awal digunakan untuk menghasilkan keluaran dasar yang dapat dianalisis sebelum dilakukan penyesuaian parameter berdasarkan hasil evaluasi.")
        scope = st.radio("Cakupan perubahan", ["Semua kamera", "Kamera tertentu"], horizontal=True, key=widget_key(case_id, "analysis", "scope"))
        camera = st.selectbox("Kamera", case["cameras"], disabled=scope == "Semua kamera", key=widget_key(case_id, "analysis", "camera")) if scope == "Kamera tertentu" else None
        source = config["camera_configs"].get(camera, config) if camera else config
        with st.form(widget_key(case_id, "analysis", "form"), border=True):
            tracking, filtering, reid = _controls(case_id, camera, source.get("tracking", config["tracking"]), source.get("filter", config["filter"]), config["reid"])
            submitted = st.form_submit_button("Terapkan Konfigurasi", icon=":material/tune:", key=widget_key(case_id, "analysis", "apply"))
        if submitted:
            if tracking["track_low_thresh"] > tracking["track_high_thresh"]:
                st.error("`track_low_thresh` tidak boleh melebihi `track_high_thresh`.")
            else:
                updated = normalize_config(_apply_scope(config, tracking, filtering, reid, scope, camera))
                message = invalidate_for_config_change(state, state.get("config"), updated, case)
                st.success(message or "Konfigurasi diterapkan.")
    with st.expander("Detail konfigurasi runtime"):
        st.json(state["config"])
    return state["config"]
