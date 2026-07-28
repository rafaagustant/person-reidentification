from __future__ import annotations

from pathlib import Path
import time
import json
import logging
import traceback

import pandas as pd
import streamlit as st

from config.cases import DEMO_CASES, get_case_by_id
from config.gt_cases import with_gt_meta
from core.models import cuda_status
from core.paths import OSNET_DEFAULT_PATH, YOLO_LOCAL_PATH
from core.pipeline import create_run_dir, render_reid_outputs, run_reid_stage, run_tracking_stage
from config.presets import normalize_config
from ui.config_panel import render_config_panel
from ui.components import dataframe_with_tools
from ui.results_panel import render_results
from ui.result_adapter import manifest_metrics
from ui.state import (
    camera_tracking_fingerprint,
    ensure_camera_tracking_state,
    get_case_state,
    init_state,
    invalidate_downstream,
    refresh_camera_tracking_state,
    set_reid_failure,
)


logger = logging.getLogger(__name__)


def _asset_rows(case: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"camera": camera, "video": str(case["video_files"][camera]), "available": Path(case["video_files"][camera]).exists()}
        for camera in case["cameras"]
    ])


def _case_overview(case: dict) -> None:
    st.subheader(case["title"])
    st.caption(case.get("description", ""))
    with st.container(horizontal=True):
        st.metric("Kamera", len(case["cameras"]), border=True)
        st.metric("Rentang frame", f"{case.get('source_frame_start', '-')}-{case.get('source_frame_end', '-')}", border=True)
        st.metric("ID ground truth", case.get("gt_identity", "N/A"), border=True)
    dataframe_with_tools(_asset_rows(case), f"{case['case_id']}_asset_status.csv", f"{case['case_id']}__case_overview__assets")


def _save_camera_manifest(state: dict, case: dict) -> None:
    run_dir = Path(state.get("run_dir") or "")
    if not run_dir.is_dir():
        return
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    statuses = state.get("camera_tracking_status", {})
    fingerprints = state.get("camera_tracking_config_fingerprints", {})
    manifest["camera_tracking"] = {
        camera: {
            "status": statuses.get(camera, "not_started"),
            "config_fingerprint": fingerprints.get(camera),
            "output_path": str(run_dir / camera),
        }
        for camera in case.get("cameras", [])
    }
    manifest.update({
        "required_cameras": case.get("cameras", []),
        "ready_cameras": [camera for camera in case.get("cameras", []) if statuses.get(camera) == "ready"],
        "tracking_complete": bool(state.get("tracking_done")),
        "aggregate_output_paths": {"valid_tracks": str(run_dir / "valid_tracks_all.csv")},
    })
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _run_tracking(case: dict, state: dict, yolo_weight: str, use_cuda: bool, cameras: list[str] | None = None) -> None:
    selected = cameras or list(case["cameras"])
    assets = _asset_rows(case)
    if not assets[assets["camera"].isin(selected)]["available"].all():
        st.error("Video case belum tersedia. Tambahkan aset sebelum menjalankan tracking.")
        return
    ensure_camera_tracking_state(case, state)
    run_dir = Path(state["run_dir"]) if state.get("run_dir") else create_run_dir(case)
    progress = st.progress(0)
    progress_text = st.empty()
    def on_progress(camera: str, done: int, total: int) -> None:
        percent = min(100, int(done / max(1, total) * 100))
        progress.progress(percent)
        progress_text.caption(f"Tracking {camera}: {done}/{total} frame ({percent}%)")
    with st.status("Menjalankan tracking lokal", expanded=True) as status:
        started = time.perf_counter()
        logger.info("Running tracking for case %s / %s", case["case_id"], ", ".join(selected))
        for camera in selected:
            state["camera_tracking_status"][camera] = "running"
        result = run_tracking_stage(case, state["config"], yolo_weight=yolo_weight, use_cuda=use_cuda, run_dir=run_dir, progress_callback=on_progress, camera_names=selected)
        progress.progress(100)
        status.update(label=f"Tracking selesai dalam {time.perf_counter() - started:.1f} detik", state="complete")
    for camera in selected:
        state["camera_tracking_results"][camera] = result.get("camera_results", {}).get(camera, {})
        state["camera_tracking_status"][camera] = "ready"
        state["camera_tracking_config_fingerprints"][camera] = camera_tracking_fingerprint(case, state["config"], camera)
        state["camera_tracking_errors"].pop(camera, None)
    state.update({"run_dir": str(run_dir), "tracking_result": result, "last_error": None, "last_error_stage": None, "last_error_message": None, "last_error_traceback": None})
    refresh_camera_tracking_state(case, state)
    invalidate_downstream(state)
    _save_camera_manifest(state, case)
    _save_run_metrics(state)


def validate_reid_inputs(case: dict, state: dict, osnet_weight: str, use_cuda: bool) -> tuple[bool, str]:
    refresh_camera_tracking_state(case, state)
    pending = [camera for camera in case.get("cameras", []) if state.get("camera_tracking_status", {}).get(camera) != "ready"]
    if pending:
        return False, f"Tracking belum siap untuk {', '.join(pending)}."
    tracking = state.get("tracking_result")
    if not state.get("tracking_done") or not tracking:
        return False, "Re-ID tidak dapat dijalankan karena hasil tracking tidak tersedia."
    run_dir = Path(state.get("run_dir") or "")
    if not state.get("run_dir") or not run_dir.is_dir():
        return False, "Re-ID tidak dapat dijalankan karena direktori output tracking tidak tersedia."
    result_run_dir = Path(tracking.get("run_dir", run_dir))
    if result_run_dir.resolve() != run_dir.resolve():
        return False, "Output tracking tidak cocok dengan run aktif."
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")).get("case_id") != case["case_id"]:
                return False, "Output tracking berasal dari case yang berbeda."
        except Exception:
            return False, "Run manifest tracking tidak dapat dibaca."
    valid_summary = tracking.get("valid_summary_all")
    valid_tracks = tracking.get("valid_tracks_all")
    if valid_summary is None or valid_tracks is None or len(valid_tracks) == 0:
        return False, "Re-ID tidak dapat dijalankan karena tidak terdapat valid track."
    crop_column = valid_tracks.get("crop_path") if "crop_path" in valid_tracks else None
    crop_count = sum(Path(str(path)).is_file() for path in crop_column) if crop_column is not None else 0
    if crop_count == 0:
        return False, "Crop valid track tidak tersedia."
    weight_path = Path(osnet_weight) if osnet_weight else None
    if weight_path is None or not weight_path.is_file():
        return False, f"Bobot OSNet tidak ditemukan: {osnet_weight}."
    try:
        with weight_path.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        return False, f"Bobot OSNet tidak dapat dibaca: {error}."
    try:
        config = normalize_config(state.get("config"))
    except Exception as error:
        return False, f"Konfigurasi Re-ID tidak valid: {error}."
    required = {"enable_cross_camera", "enable_strict_intra", "use_mnn", "cross_threshold", "intra_threshold", "intra_max_gap", "intra_max_overlap"}
    if not required.issubset(config.get("reid", {})):
        return False, "Konfigurasi Re-ID tidak lengkap."
    if use_cuda is not True and use_cuda is not False:
        return False, "Device Re-ID tidak valid."
    return True, ""


def _run_reid(case: dict, state: dict, osnet_weight: str, use_cuda: bool) -> None:
    valid, message = validate_reid_inputs(case, state, osnet_weight, use_cuda)
    if not valid:
        raise ValueError(message)
    state.update({"reid_status": "running", "reid_done": False, "reid_result": None, "render_result": None, "render_done": False})
    progress = st.progress(0)
    progress_text = st.empty()
    def on_progress(done: int, total: int) -> None:
        percent = min(70, int(done / max(1, total) * 70))
        progress.progress(percent)
        progress_text.caption(f"Ekstraksi embedding: {done}/{total} crop ({percent}%)")
    with st.status("Menjalankan OSNet dan asosiasi Global ID", expanded=True) as status:
        result = run_reid_stage(case, state["config"], state["run_dir"], osnet_weight=osnet_weight, use_cuda=use_cuda, reid_progress_callback=on_progress, render=False)
        progress.progress(100)
        status.update(label="Re-ID selesai", state="complete")
    state.update({"reid_result": result, "reid_done": True, "reid_status": "completed", "render_done": False, "render_result": None, "last_error": None, "last_error_stage": None, "last_error_message": None, "last_error_traceback": None})
    _save_run_metrics(state)


def _save_run_metrics(state: dict) -> None:
    run_dir = Path(state["run_dir"])
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(manifest_metrics(state.get("tracking_result"), state.get("reid_result")))
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _render(case: dict, state: dict) -> None:
    progress = st.progress(0)
    progress_text = st.empty()
    def on_progress(camera: str, done: int, total: int) -> None:
        percent = min(100, int(done / max(1, total) * 100))
        progress.progress(percent)
        progress_text.caption(f"Render {camera}: {done}/{total} frame ({percent}%)")
    with st.status("Merender video hasil", expanded=True) as status:
        rendered = render_reid_outputs(case, state["run_dir"], render_df=state["reid_result"].get("render_df"), render_progress_callback=on_progress)
        progress.progress(100)
        state["reid_result"].update(rendered)
        state.update({"render_result": rendered, "render_done": True, "last_error": None})
        status.update(label="Render selesai", state="complete")


def _camera_tracking_table(case: dict, state: dict) -> pd.DataFrame:
    labels = {"not_started": "Belum dijalankan", "running": "Sedang diproses", "ready": "Siap", "stale": "Perlu dijalankan ulang", "failed": "Gagal"}
    rows = []
    for camera in case["cameras"]:
        status = state.get("camera_tracking_status", {}).get(camera, "not_started")
        current = camera_tracking_fingerprint(case, state.get("config") or {}, camera)
        matching = state.get("camera_tracking_config_fingerprints", {}).get(camera) == current
        rows.append({
            "Kamera": camera,
            "Status Tracking": labels.get(status, status),
            "Konfigurasi": state.get("configuration_mode", "-"),
            "Fingerprint Cocok": "Ya" if status == "ready" and matching else "Tidak",
            "Siap untuk Re-ID": "Ya" if status == "ready" and matching else "Tidak",
        })
    return pd.DataFrame(rows)


def _case_page(case: dict, cuda_active: bool, yolo_weight: str, osnet_weight: str) -> None:
    case = with_gt_meta(case)
    state = get_case_state(case["case_id"])
    ensure_camera_tracking_state(case, state)
    refresh_camera_tracking_state(case, state)
    _case_overview(case)
    st.subheader("Konfigurasi")
    render_config_panel(case, state)
    st.subheader("Proses")
    is_multi = len(case["cameras"]) > 1
    if is_multi:
        st.markdown("**Tracking per Kamera**")
        dataframe_with_tools(_camera_tracking_table(case, state), f"{case['case_id']}_camera_tracking_status.csv", f"{case['case_id']}__tracking_process__status")
        selected_camera = st.selectbox("Kamera yang Dijalankan", case["cameras"], key=f"{case['case_id']}__tracking_process__selected_camera")
        left, right = st.columns(2)
        if left.button("Jalankan Kamera Terpilih", type="primary", icon=":material/videocam:", key=f"{case['case_id']}__tracking_process__run_selected_camera"):
            try:
                _run_tracking(case, state, yolo_weight, cuda_active, [selected_camera])
            except Exception as error:
                state["camera_tracking_status"][selected_camera] = "failed"
                state["camera_tracking_errors"][selected_camera] = traceback.format_exc()
                st.error(f"Tracking {selected_camera} gagal.")
        if right.button("Jalankan Semua Kamera", icon=":material/video_library:", key=f"{case['case_id']}__tracking_process__run_all_cameras"):
            pending = [camera for camera in case["cameras"] if state["camera_tracking_status"].get(camera) != "ready"]
            if not pending:
                st.info("Seluruh kamera sudah siap. Tidak ada tracking yang perlu dijalankan ulang.")
            for index, camera in enumerate(pending, start=1):
                st.caption(f"Kamera {index} dari {len(pending)}: {camera}")
                try:
                    _run_tracking(case, state, yolo_weight, cuda_active, [camera])
                except Exception:
                    state["camera_tracking_status"][camera] = "failed"
                    state["camera_tracking_errors"][camera] = traceback.format_exc()
                    st.error(f"Tracking {camera} gagal.")
        ready_count = sum(state["camera_tracking_status"].get(camera) == "ready" for camera in case["cameras"])
        st.caption(f"Status: {ready_count} dari {len(case['cameras'])} kamera siap")
        for camera, error_text in state.get("camera_tracking_errors", {}).items():
            if error_text:
                with st.expander(f"Detail error tracking {camera}"):
                    st.code(error_text, language="text")
    with st.container(horizontal=True):
        if not is_multi and st.button("Tahap 1: Tracking lokal", type="primary", icon=":material/videocam:", key=f"{case['case_id']}_tracking"):
            try:
                _run_tracking(case, state, yolo_weight, cuda_active)
            except Exception as error:
                state["camera_tracking_status"][case["cameras"][0]] = "failed"
                state["camera_tracking_errors"][case["cameras"][0]] = traceback.format_exc()
                st.error("Tracking gagal. Periksa aset dan konfigurasi.")
        pending = [camera for camera in case["cameras"] if state["camera_tracking_status"].get(camera) != "ready"]
        reid_help = f"Tracking belum siap untuk {', '.join(pending)}." if pending else None
        if reid_help:
            st.caption(reid_help)
        if st.button("Tahap 2: Re-ID dan Global ID", disabled=not state.get("tracking_done"), icon=":material/person_search:", key=f"{case['case_id']}_reid"):
            try:
                _run_reid(case, state, osnet_weight, cuda_active)
            except Exception as error:
                logger.exception("Re-ID stage failed for case %s", case["case_id"])
                set_reid_failure(state, str(error), traceback.format_exc())
        if st.button("Tahap 3: Render hasil", disabled=not state.get("reid_done"), icon=":material/movie:", key=f"{case['case_id']}_render"):
            try:
                _render(case, state)
            except Exception as error:
                state["last_error"] = str(error)
                st.error("Render gagal. Periksa video sumber dan codec.")
                with st.expander("Detail error"):
                    st.exception(error)
    if state.get("has_tracking_results") or state.get("tracking_done") or state.get("reid_done"):
        if state.get("has_tracking_results") and not state.get("tracking_done"):
            st.info(f"Hasil tracking sementara: {sum(state['camera_tracking_status'].get(camera) == 'ready' for camera in case['cameras'])} dari {len(case['cameras'])} kamera siap.")
        render_results(case, state)
    if state.get("last_error_stage") == "reid":
        st.error(f"Re-ID gagal: {state.get('last_error_message') or 'terjadi kesalahan.'}")
        with st.expander("Detail error"):
            st.code(state.get("last_error_traceback", "Tidak ada traceback."), language="text")


def main() -> None:
    st.set_page_config(page_title="Person Re-Identification Multi-Kamera", page_icon=":material/person_search:", layout="wide")
    init_state()
    st.title("Person Re-Identification Multi-Kamera")
    cuda_info = cuda_status()
    cuda_active = bool(cuda_info["cuda_available"])
    cuda_label = cuda_info["device_name"] or "CPU"
    st.caption("YOLO11n → BoT-SORT → OSNet → Asosiasi Track → Global ID")
    with st.container(horizontal=True):
        st.metric("Perangkat", cuda_label, border=True)
        st.metric("CUDA", "Aktif" if cuda_active else "Tidak aktif", border=True)
        st.metric("Bobot YOLO", "Tersedia" if Path(YOLO_LOCAL_PATH).exists() else "Tidak tersedia", border=True)
        st.metric("Bobot OSNet", "Tersedia" if Path(OSNET_DEFAULT_PATH).exists() else "Tidak tersedia", border=True)

    yolo_weight, osnet_weight = str(YOLO_LOCAL_PATH), str(OSNET_DEFAULT_PATH)
    case_tabs = st.tabs([case["title"].split(" - ")[0] for case in DEMO_CASES])
    for tab, case in zip(case_tabs, DEMO_CASES):
        with tab:
            _case_page(case, cuda_active, yolo_weight, osnet_weight)


if __name__ == "__main__":
    main()
