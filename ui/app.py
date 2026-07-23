from __future__ import annotations

from pathlib import Path
import time
import json

import pandas as pd
import streamlit as st

from config.cases import DEMO_CASES, get_case_by_id
from config.gt_cases import with_gt_meta
from core.models import cuda_status
from core.paths import OSNET_DEFAULT_PATH, YOLO_LOCAL_PATH
from core.pipeline import create_run_dir, render_reid_outputs, run_reid_stage, run_tracking_stage
from ui.config_panel import render_config_panel
from ui.results_panel import render_results
from ui.state import get_case_state, init_state


def _asset_rows(case: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"camera": camera, "video": str(case["video_files"][camera]), "available": Path(case["video_files"][camera]).exists()}
        for camera in case["cameras"]
    ])


def _case_selector() -> dict:
    labels = {case["case_id"]: case["title"] for case in DEMO_CASES}
    case_id = st.selectbox("Case pengujian", list(labels), format_func=labels.get)
    return with_gt_meta(get_case_by_id(case_id))


def _case_overview(case: dict) -> None:
    st.subheader(case["title"])
    st.caption(case.get("description", ""))
    with st.container(horizontal=True):
        st.metric("Kamera", len(case["cameras"]), border=True)
        st.metric("Rentang frame", f"{case.get('source_frame_start', '-')}-{case.get('source_frame_end', '-')}", border=True)
        st.metric("ID ground truth", case.get("gt_identity", "N/A"), border=True)
    st.dataframe(_asset_rows(case), hide_index=True)


def _run_tracking(case: dict, state: dict, yolo_weight: str, use_cuda: bool) -> None:
    if not all(_asset_rows(case)["available"]):
        st.error("Video case belum tersedia. Tambahkan aset sebelum menjalankan tracking.")
        return
    run_dir = create_run_dir(case)
    with st.status("Menjalankan tracking lokal", expanded=True) as status:
        started = time.perf_counter()
        result = run_tracking_stage(case, state["config"], yolo_weight=yolo_weight, use_cuda=use_cuda, run_dir=run_dir)
        status.update(label=f"Tracking selesai dalam {time.perf_counter() - started:.1f} detik", state="complete")
    state.update({"run_dir": str(run_dir), "tracking_result": result, "tracking_done": True, "reid_done": False, "reid_result": None})
    _save_run_metrics(state)


def _run_reid(case: dict, state: dict, osnet_weight: str, use_cuda: bool) -> None:
    with st.status("Menjalankan OSNet dan asosiasi Global ID", expanded=True) as status:
        result = run_reid_stage(case, state["config"], state["run_dir"], osnet_weight=osnet_weight, use_cuda=use_cuda, render=False)
        status.update(label="Re-ID selesai", state="complete")
    state.update({"reid_result": result, "reid_done": True})
    _save_run_metrics(state)


def _save_run_metrics(state: dict) -> None:
    run_dir = Path(state["run_dir"])
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tracking_metrics = (state.get("tracking_result") or {}).get("standard_metrics_df", pd.DataFrame())
    reid_metrics = (state.get("reid_result") or {}).get("reid_pairwise_eval_df", pd.DataFrame())
    manifest["tracking_metrics"] = tracking_metrics.iloc[0].to_dict() if len(tracking_metrics) else {}
    manifest["global_id_metrics"] = reid_metrics.iloc[0].to_dict() if len(reid_metrics) else {}
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _render(case: dict, state: dict) -> None:
    with st.status("Merender video hasil", expanded=True) as status:
        rendered = render_reid_outputs(case, state["run_dir"], render_df=state["reid_result"].get("render_df"))
        state["reid_result"].update(rendered)
        status.update(label="Render selesai", state="complete")


def main() -> None:
    st.set_page_config(page_title="Demo Person Re-ID", layout="wide")
    init_state()
    st.title("Demonstrasi person re-identification")
    cuda_info = cuda_status()
    cuda_active = bool(cuda_info["cuda_available"])
    cuda_label = cuda_info["device_name"] or "CPU"
    st.caption(f"YOLO11n + BoT-SORT local tracking + OSNet track-level Re-ID. Perangkat: {cuda_label}.")

    case = _case_selector()
    state = get_case_state(case["case_id"])
    _case_overview(case)
    st.subheader("Konfigurasi")
    render_config_panel(case, state)

    yolo_weight, osnet_weight = str(YOLO_LOCAL_PATH), str(OSNET_DEFAULT_PATH)
    models_available = Path(yolo_weight).exists() and Path(osnet_weight).exists()
    st.caption("Model dan aset: tersedia" if models_available else "Model belum lengkap; tombol proses dapat gagal sampai bobot tersedia.")
    st.subheader("Proses")
    with st.container(horizontal=True):
        if st.button("Jalankan tracking", type="primary", icon=":material/videocam:"):
            try:
                _run_tracking(case, state, yolo_weight, cuda_active)
            except Exception as error:
                st.exception(error)
        if st.button("Jalankan Re-ID", disabled=not state.get("tracking_done"), icon=":material/person_search:"):
            try:
                _run_reid(case, state, osnet_weight, cuda_active)
            except Exception as error:
                st.exception(error)
        if st.button("Render hasil", disabled=not state.get("reid_done"), icon=":material/movie:"):
            try:
                _render(case, state)
            except Exception as error:
                st.exception(error)
    if state.get("tracking_done") or state.get("reid_done"):
        render_results(case, state)


if __name__ == "__main__":
    main()
