from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.paths import OUTPUT_ROOT


def _first_row(value: pd.DataFrame | None) -> dict:
    return value.iloc[0].to_dict() if value is not None and len(value) else {}


def _metric(value: object) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.3f}" if isinstance(value, float) else str(value)


def _previous_manifest(case_id: str, current_run: str | None) -> dict | None:
    root = OUTPUT_ROOT / case_id
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and str(path) != str(current_run)]
    if not candidates:
        return None
    manifest = max(candidates, key=lambda path: path.stat().st_mtime) / "run_manifest.json"
    if not manifest.exists():
        return None
    try:
        import json
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_results(case: dict, state: dict) -> None:
    tracking = state.get("tracking_result") or {}
    reid = state.get("reid_result") or {}
    summary, video, gallery, analysis = st.tabs(["Ringkasan", "Video", "Gallery", "Analisis"])
    with summary:
        tracking_metrics = _first_row(tracking.get("standard_metrics_df"))
        reid_metrics = _first_row(reid.get("reid_pairwise_eval_df"))
        with st.container(horizontal=True):
            st.metric("Raw track", _metric(_first_row(tracking.get("tracking_eval_df")).get("unique_tracks")), border=True)
            st.metric("Valid track", _metric(len(reid.get("track_embedding_df", []))), border=True)
            st.metric("Global ID", _metric(len(reid.get("global_meta_df", []))), border=True)
            st.metric("Merged pair", _metric(int(reid.get("pair_df", pd.DataFrame()).get("merge_status", pd.Series(dtype=bool)).sum())), border=True)
        with st.container(horizontal=True):
            st.metric("Precision", _metric(tracking_metrics.get("precision")), border=True)
            st.metric("Recall", _metric(tracking_metrics.get("recall")), border=True)
            st.metric("F1", _metric(tracking_metrics.get("f1")), border=True)
            st.metric("MOTA", _metric(tracking_metrics.get("mota_simple")), border=True)
            st.metric("Pairwise F1", _metric(reid_metrics.get("pairwise_f1")), border=True)
        st.subheader("Konfigurasi yang digunakan")
        st.json(state.get("config", {}), expanded=False)
    with video:
        for path in reid.get("rendered_paths", []):
            if Path(path).exists():
                st.video(str(path))
        if reid.get("combined_path") and Path(reid["combined_path"]).exists():
            st.video(str(reid["combined_path"]))
    with gallery:
        gallery_df = reid.get("global_gallery_df", pd.DataFrame())
        if len(gallery_df):
            st.dataframe(gallery_df, hide_index=True)
        else:
            st.info("Gallery tersedia setelah Re-ID selesai.")
    with analysis:
        pairs = reid.get("pair_df", pd.DataFrame())
        if len(pairs):
            st.dataframe(pairs.sort_values("cosine_similarity", ascending=False), hide_index=True)
        hints = (tracking.get("tracking_tuning_hints") or []) + (reid.get("reid_tuning_hints") or [])
        if hints:
            st.subheader("Saran analisis kesalahan")
            st.dataframe(pd.DataFrame(hints), hide_index=True)
        previous = _previous_manifest(case["case_id"], state.get("run_dir"))
        if previous:
            current_tracking = _first_row(tracking.get("standard_metrics_df"))
            current_reid = _first_row(reid.get("reid_pairwise_eval_df"))
            prior_tracking = previous.get("tracking_metrics", {})
            prior_reid = previous.get("global_id_metrics", {})
            metric_keys = ["precision", "recall", "f1", "mota_simple", "id_switch_count", "fragmentation_avg", "pairwise_f1", "false_merge_rate", "false_split_rate", "global_id_purity"]
            rows = []
            for key in metric_keys:
                before = prior_tracking.get(key, prior_reid.get(key))
                after = current_tracking.get(key, current_reid.get(key))
                if before is not None or after is not None:
                    delta = float(after) - float(before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
                    rows.append({"metrik": key, "run sebelumnya": before, "run saat ini": after, "selisih": delta})
            st.subheader("Perbandingan dengan run sebelumnya")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
