from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import streamlit as st

from core.paths import OUTPUT_ROOT


def init_state():
    if "case_states" not in st.session_state:
        st.session_state["case_states"] = {}


def get_case_state(case_id: str) -> dict:
    init_state()
    if case_id not in st.session_state["case_states"]:
        st.session_state["case_states"][case_id] = {
            "selected_case_id": case_id,
            "configuration_mode": "Konfigurasi Rekomendasi",
            "tracking_done": False,
            "reid_done": False,
            "render_done": False,
            "run_dir": None,
            "config": None,
            "config_fingerprint": None,
            "tracking_result": None,
            "reid_result": None,
            "render_result": None,
            "last_error": None,
            "last_error_stage": None,
            "last_error_message": None,
            "last_error_traceback": None,
            "reid_status": "not_started",
            "progress_state": {},
            "camera_tracking_results": {},
            "camera_tracking_status": {},
            "camera_tracking_config_fingerprints": {},
            "camera_tracking_errors": {},
            "has_tracking_results": False,
        }
    return st.session_state["case_states"][case_id]


def camera_tracking_fingerprint(case: dict, config: dict, camera: str) -> str:
    camera_config = (config.get("camera_configs") or {}).get(camera, {})
    payload = {
        "camera": camera,
        "video": str((case.get("video_files") or {}).get(camera, "")),
        "source_frame_start": case.get("source_frame_start"),
        "source_frame_end": case.get("source_frame_end"),
        "tracking": camera_config.get("tracking", config.get("tracking", {})),
        "filter": camera_config.get("filter", config.get("filter", {})),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def ensure_camera_tracking_state(case: dict, state: dict) -> None:
    statuses = state.setdefault("camera_tracking_status", {})
    results = state.setdefault("camera_tracking_results", {})
    fingerprints = state.setdefault("camera_tracking_config_fingerprints", {})
    state.setdefault("camera_tracking_errors", {})
    for camera in case.get("cameras", []):
        statuses.setdefault(camera, "not_started")
        if camera in results and camera not in fingerprints:
            fingerprints[camera] = camera_tracking_fingerprint(case, state.get("config") or {}, camera)
            statuses[camera] = "ready"


def refresh_camera_tracking_state(case: dict, state: dict) -> None:
    ensure_camera_tracking_state(case, state)
    config = state.get("config") or {}
    for camera in case.get("cameras", []):
        current = camera_tracking_fingerprint(case, config, camera)
        if state["camera_tracking_status"].get(camera) == "ready" and state["camera_tracking_config_fingerprints"].get(camera) != current:
            state["camera_tracking_status"][camera] = "stale"
    ready = [camera for camera in case.get("cameras", []) if state["camera_tracking_status"].get(camera) == "ready"]
    state["has_tracking_results"] = bool(ready)
    state["tracking_done"] = len(ready) == len(case.get("cameras", []))


def invalidate_downstream(state: dict) -> None:
    state.update({"reid_result": None, "render_result": None, "reid_done": False, "render_done": False, "reid_status": "not_started"})


def config_fingerprints(config: dict) -> tuple[str, str]:
    tracking_payload = {
        "tracking": config.get("tracking", {}),
        "filter": config.get("filter", {}),
        "camera_configs": config.get("camera_configs", {}),
    }
    reid_payload = config.get("reid", {})
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest(tracking_payload), digest(reid_payload)


def invalidate_for_config_change(state: dict, before: dict | None, after: dict, case: dict | None = None) -> str | None:
    before_tracking, before_reid = config_fingerprints(before or {})
    after_tracking, after_reid = config_fingerprints(after)
    state["config"] = after
    state["config_fingerprint"] = {"tracking": after_tracking, "reid": after_reid}
    if before and before_tracking != after_tracking:
        if case is not None:
            ensure_camera_tracking_state(case, state)
            changed = []
            for camera in case.get("cameras", []):
                if camera_tracking_fingerprint(case, before, camera) != camera_tracking_fingerprint(case, after, camera):
                    state["camera_tracking_status"][camera] = "stale"
                    changed.append(camera)
            invalidate_downstream(state)
            refresh_camera_tracking_state(case, state)
            if len(changed) == 1:
                return f"Konfigurasi tracking {changed[0]} berubah. Jalankan ulang tracking kamera tersebut."
            return "Konfigurasi tracking berubah. Jalankan ulang kamera yang perlu diperbarui."
        state.update({"tracking_result": None, "reid_result": None, "render_result": None, "tracking_done": False, "reid_done": False, "render_done": False, "reid_status": "not_started", "last_error_stage": None, "last_error_message": None, "last_error_traceback": None})
        return "Parameter tracking berubah. Tracking perlu dijalankan ulang."
    if before and before_reid != after_reid:
        invalidate_downstream(state)
        state.update({"last_error_stage": None, "last_error_message": None, "last_error_traceback": None})
        return "Parameter Re-ID berubah. Hasil tracking tetap digunakan, tetapi Re-ID perlu dijalankan ulang."
    return None


def set_reid_failure(state: dict, message: str, traceback_text: str) -> None:
    state.update({
        "reid_result": None,
        "render_result": None,
        "reid_done": False,
        "render_done": False,
        "reid_status": "failed",
        "last_error": message,
        "last_error_stage": "reid",
        "last_error_message": message,
        "last_error_traceback": traceback_text,
    })


def reset_case(case_id: str, delete_output: bool = False):
    state = get_case_state(case_id)
    run_dir = state.get("run_dir")
    if delete_output and run_dir and Path(run_dir).exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    st.session_state["case_states"].pop(case_id, None)


def reset_all(delete_output: bool = False):
    if delete_output:
        for state in st.session_state.get("case_states", {}).values():
            run_dir = state.get("run_dir")
            if run_dir and Path(run_dir).exists():
                shutil.rmtree(run_dir, ignore_errors=True)
    st.session_state["case_states"] = {}


def clear_current_run_output(case_id: str) -> bool:
    state = get_case_state(case_id)
    run_dir = state.get("run_dir")
    if not run_dir:
        return False
    path = Path(run_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    st.session_state["case_states"].pop(case_id, None)
    return True


def clear_selected_case_outputs(case_id: str) -> bool:
    case_dir = OUTPUT_ROOT / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    st.session_state.get("case_states", {}).pop(case_id, None)
    return True


def clear_all_old_runs(keep_latest: int = 3) -> int:
    removed = 0
    root = OUTPUT_ROOT
    if not root.exists():
        return removed
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        runs = sorted(
            [p for p in case_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in runs[int(keep_latest):]:
            shutil.rmtree(run_dir, ignore_errors=True)
            removed += 1
    return removed


def clear_streamlit_cache_and_state() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["case_states"] = {}
