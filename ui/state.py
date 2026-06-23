from __future__ import annotations

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
            "tracking_done": False,
            "reid_done": False,
            "run_dir": None,
            "config": None,
            "tracking_result": None,
            "reid_result": None,
        }
    return st.session_state["case_states"][case_id]


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
