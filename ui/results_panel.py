from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import crop_gallery, dataframe_with_tools, filtered_track_gallery, global_id_gallery, merged_track_gallery
from ui.result_adapter import format_count, format_ratio, metric_status, reid_view, tracking_view


_PAIR_LABELS = {
    "track_a": "Track A", "camera_a": "Kamera A", "track_b": "Track B", "camera_b": "Kamera B",
    "frame_range_a": "Rentang Frame A", "frame_range_b": "Rentang Frame B",
    "overlap_frame_count": "Overlap Frame", "temporal_gap": "Temporal Gap", "temporal_status": "Status Temporal",
    "cosine_similarity": "Cosine Similarity", "threshold": "Threshold", "similarity_margin": "Similarity Margin",
    "association_type": "Tipe Asosiasi", "mnn_status": "MNN", "decision": "Keputusan",
    "rejection_reason": "Alasan Penolakan", "resulting_global_id": "Global ID Hasil",
}


def _pair_display(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame]].rename(columns=_PAIR_LABELS)


def _metric(label: str, value, *, ratio: bool = False, status: str = "") -> None:
    st.metric(label, format_ratio(value, status) if ratio else format_count(value, status), border=True)


def render_results(case: dict, state: dict) -> None:
    tracking = tracking_view(state.get("tracking_result"))
    reid = reid_view(state.get("reid_result"))
    summary_tab, tracking_tab, reid_tab, gallery_tab, video_tab = st.tabs(["Ringkasan", "Tracking", "Re-ID dan Global ID", "Gallery", "Video"])
    metrics = tracking["metrics"]
    with summary_tab:
        with st.container(horizontal=True):
            _metric("Raw Track", tracking["raw_track_count"])
            _metric("Valid Track", tracking["valid_track_count"])
            _metric("Track Terfilter", tracking["filtered_track_count"])
            for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("mota", "MOTA")]:
                _metric(label, metrics.get(key), ratio=True, status=metric_status(metrics.get(key), gt_available=tracking["gt_available"]))
        st.caption(
            "Raw Track: seluruh local track sebelum filter. "
            "Valid Track: local track yang lolos filter. "
            "Track Terfilter: local track yang tidak lolos filter."
        )
        if state.get("reid_done"):
            reid_metrics = reid["metrics"]
            with st.container(horizontal=True):
                _metric("Global ID", reid["global_ids"])
                _metric("Merged pair", reid["merged_pairs"])
                positive_pairs = reid_metrics.get("positive_pair_count", 0)
                _metric("Pairwise F1", reid_metrics.get("pairwise_f1"), ratio=True, status=metric_status(reid_metrics.get("pairwise_f1"), applicable=bool(positive_pairs)))
                _metric("False Merge Rate", reid["false_merge_rate"], ratio=True)
                _metric("False Split Rate", reid["false_split_rate"], ratio=True)
                _metric("Global ID purity", reid_metrics.get("global_id_purity"), ratio=True, status=metric_status(reid_metrics.get("global_id_purity")))
            st.caption(f"Proses Re-ID membentuk {reid['global_ids']} Global ID dari {format_count(tracking['valid_track_count'])} valid track dan menggabungkan {reid['merged_pairs']} pasangan track.")
        else:
            st.info(
                "Tracking menghasilkan "
                f"{format_count(tracking['valid_track_count'])} valid track dari "
                f"{format_count(tracking['raw_track_count'])} raw track. "
                "Metrik Re-ID belum dihitung."
            )
    with tracking_tab:
        display_camera = None
        ready_cameras = [camera for camera, status in state.get("camera_tracking_status", {}).items() if status == "ready"]
        if len(case.get("cameras", [])) > 1 and ready_cameras:
            display_camera = st.selectbox(
                "Kamera Hasil Tracking",
                ["Semua kamera siap", *ready_cameras],
                key=f"{case['case_id']}__tracking_results__camera_filter",
            )

        def camera_view(frame: pd.DataFrame) -> pd.DataFrame:
            if display_camera and display_camera != "Semua kamera siap" and "camera" in frame:
                return frame[frame["camera"] == display_camera]
            return frame

        detail = {key: metrics.get(key) for key in ["tp", "fp", "fn", "id_switch_count", "fragmentation_avg", "gt_coverage_rate", "pred_match_rate"] if key in metrics}
        dataframe_with_tools(pd.DataFrame([detail]), f"{case['case_id']}_tracking_detail.csv", f"{case['case_id']}__tracking__detail")
        for title, frame, columns in [
            ("Jumlah track per kamera", tracking["track_count_by_camera_df"], ["camera", "raw_track_count", "valid_track_count", "filtered_track_count"]),
            ("Hasil per kamera", tracking["standard_by_camera_df"], ["camera", "precision", "recall", "f1", "mota", "id_switch_count", "fragmentation_avg"]),
            ("Status kamera", tracking["camera_status_df"], ["camera", "status", "raw_track_count", "valid_track_count", "filtered_track_count", "runtime_sec"]),
            ("Ringkasan local track", tracking["valid_summary_df"], ["camera", "track_id", "track_key", "num_frames", "num_crops", "avg_conf", "avg_area", "status", "filter_reason"]),
        ]:
            frame = camera_view(frame)
            if len(frame):
                st.subheader(title)
                displayed = frame[[col for col in columns if col in frame]]
                table_key = f"{case['case_id']}__tracking__{title.lower().replace(' ', '_')}"
                dataframe_with_tools(displayed, f"{case['case_id']}_{title.lower().replace(' ', '_')}.csv", table_key)
        st.subheader("Filter Diagnostics")
        with st.expander("Filtered Track Gallery"):
            filtered = camera_view(tracking["filtered_tracks_df"])
            if not state.get("tracking_done"):
                st.info("Jalankan tracking untuk menampilkan Filtered Track Gallery.")
            elif not len(filtered):
                st.info("Tidak terdapat track yang terfilter.")
            else:
                reason = st.selectbox("Alasan filter", ["Semua", "min_frames", "min_crops", "min_avg_conf", "min_avg_area"], key=f"{case['case_id']}__filtered_track_gallery__reason_filter")
                shown = filtered if reason == "Semua" else filtered[filtered["filter_reasons"].str.contains(reason, na=False)]
                if len(shown):
                    filtered_track_gallery(shown)
                else:
                    st.info("Tidak terdapat track terfilter dengan alasan tersebut.")
        coverage = camera_view(tracking["temporal_coverage_df"])
        st.subheader("Analisis Cakupan Temporal Ground Truth per Identitas")
        if not tracking["gt_available"]:
            st.info("GT tidak tersedia.")
        elif not len(coverage):
            st.info("Tidak terdapat identitas ground truth pada rentang pengujian.")
        else:
            weighted = coverage["matched_frame_count"].sum() / max(1, coverage["gt_frame_count"].sum())
            with st.container(horizontal=True):
                _metric("Tercakup penuh", int((coverage["coverage_status"] == "Tercakup penuh").sum()))
                _metric("Tercakup sebagian", int((coverage["coverage_status"] == "Tercakup sebagian").sum()))
                _metric("Tidak ditemukan", int((coverage["coverage_status"] == "Tidak ditemukan").sum()))
                _metric("Temporal coverage", weighted, ratio=True)
                _metric("GT terfragmentasi", int((coverage["num_matched_tracks"] > 1).sum()))
            dataframe_with_tools(coverage, f"{case['case_id']}_gt_temporal_coverage.csv", f"{case['case_id']}__gt_temporal_coverage")
    with reid_tab:
        if not state.get("reid_done"):
            st.info("Belum dihitung. Jalankan Re-ID setelah tracking selesai.")
        else:
            dataframe_with_tools(pd.DataFrame([reid["metrics"]]).drop(columns=["note"], errors="ignore"), f"{case['case_id']}_reid_metrics.csv", f"{case['case_id']}__reid__metrics")
            if len(reid["global_meta_df"]):
                st.subheader("Global ID")
                dataframe_with_tools(reid["global_meta_df"], f"{case['case_id']}_global_id.csv", f"{case['case_id']}__reid__global_id")
            if len(reid["compact_pairs_df"]):
                pairs = reid["compact_pairs_df"]
                st.subheader("Pair Similarity")
                core_cols = [
                    "track_a", "camera_a", "track_b", "camera_b",
                    "frame_range_a", "frame_range_b", "overlap_frame_count", "temporal_gap", "temporal_status",
                    "cosine_similarity", "threshold", "similarity_margin", "association_type", "mnn_status", "decision",
                ]
                pair_display = _pair_display(pairs, core_cols)
                dataframe_with_tools(pair_display, f"{case['case_id']}_pair_similarity.csv", f"{case['case_id']}__pair_similarity")
                threshold = state["config"]["reid"].get("cross_threshold", 0.75)
                nearby = pairs.loc[(pairs["cosine_similarity"] - threshold).abs() <= 0.03].copy()
                st.subheader("Pasangan di sekitar threshold")
                near_cols = [
                    "track_a", "camera_a", "track_b", "camera_b", "cosine_similarity", "threshold",
                    "similarity_margin", "frame_range_a", "frame_range_b", "overlap_frame_count", "temporal_gap", "temporal_status",
                    "association_type", "mnn_status", "decision", "rejection_reason",
                ]
                near_display = _pair_display(nearby, [col for col in near_cols if col != "rejection_reason" or nearby[col].astype(bool).any()])
                dataframe_with_tools(near_display, f"{case['case_id']}_near_threshold_pairs.csv", f"{case['case_id']}__near_threshold_pairs")
                st.subheader("Accepted Pairs")
                accepted_cols = [
                    "track_a", "track_b", "frame_range_a", "frame_range_b", "overlap_frame_count", "temporal_gap", "cosine_similarity", "threshold",
                    "association_type", "mnn_status", "resulting_global_id",
                ]
                accepted = reid["accepted_pairs_df"].copy()
                accepted_display = _pair_display(accepted, accepted_cols)
                dataframe_with_tools(accepted_display, f"{case['case_id']}_accepted_pairs.csv", f"{case['case_id']}__accepted_pairs")
                st.subheader("Rejected Pairs")
                rejected = reid["rejected_pairs_df"]
                rejected_cols = [
                    "track_a", "track_b", "frame_range_a", "frame_range_b", "overlap_frame_count", "temporal_gap", "cosine_similarity", "threshold",
                    "association_type", "mnn_status",
                ] + (["rejection_reason"] if "rejection_reason" in rejected and rejected["rejection_reason"].astype(bool).any() else [])
                rejected_display = _pair_display(rejected, rejected_cols)
                dataframe_with_tools(rejected_display, f"{case['case_id']}_rejected_pairs.csv", f"{case['case_id']}__rejected_pairs")
    with gallery_tab:
        local_tab, global_tab, merged_tab = st.tabs(["Local Track", "Global ID", "Merged Track"])
        with local_tab:
            crop_gallery(tracking["local_gallery_df"], "Local Track Gallery", f"{case['case_id']}_local_gallery")
        with global_tab:
            global_id_gallery(reid["global_id_gallery_cards_df"])
        with merged_tab:
            if not state.get("reid_done"):
                st.info("Jalankan Re-ID untuk menampilkan Merged Track Gallery.")
            elif len(reid["merged_global_gallery_df"]):
                merged_track_gallery(reid["merged_global_gallery_df"], reid["merged_global_gallery_members_df"])
            else:
                st.info("Belum terdapat Global ID yang terbentuk dari penggabungan beberapa local track.")
    with video_tab:
        st.subheader("Video Hasil")
        st.caption(case["title"])
        video_path = reid["combined_path"]
        if not video_path and len(case.get("cameras", [])) == 1 and len(reid["rendered_paths"]) == 1:
            video_path = reid["rendered_paths"][0]
        if video_path and Path(video_path).exists():
            st.video(str(video_path))
        elif len(case.get("cameras", [])) > 1:
            st.info("Video gabungan belum tersedia. Jalankan tahap render.")
        else:
            st.info("Video hasil belum tersedia. Jalankan tahap render.")
