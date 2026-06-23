from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

from core.video_io import get_video_info, make_preview_grid
try:
    from config.gt_cases import GT_CASE_META
except Exception:
    GT_CASE_META = {}


def hero():
    st.title("Person Re-Identification Multi-Kamera")
    st.caption("YOLO11n + BoT-SORT + OSNet untuk pembentukan Global ID dan evaluasi ground truth.")


def case_info(case: dict):
    st.subheader(case["title"])
    st.write(case.get("description", ""))
    gt_meta = GT_CASE_META.get(case.get("case_id"), {})
    rec_text = case.get("recommended_config", "Recommended config final")

    cols = st.columns(5)
    cols[0].metric("Jumlah kamera", len(case.get("cameras", [])))
    cols[1].metric("Kamera", ", ".join(case.get("cameras", [])))
    cols[2].metric("Skenario", case.get("scene_type", "-"))
    cols[3].metric("Frame GT", f"{gt_meta.get('source_frame_start', '-')}-{gt_meta.get('source_frame_end', '-')}")
    cols[4].metric("Rekomendasi case", rec_text)
    if case.get("expected_result"):
        st.info(case.get("expected_result", ""))


def asset_status(case: dict):
    rows = []
    for cam in case["cameras"]:
        video_path = Path(case["video_files"][cam])
        gt_meta = GT_CASE_META.get(case.get("case_id"), {})
        annotation_dir = Path(gt_meta.get("annotation_dir", "")) if gt_meta.get("annotation_dir") else None
        rows.append({
            "camera": cam,
            "video_path": str(video_path),
            "video_exists": video_path.exists(),
            "annotation_dir": str(annotation_dir) if annotation_dir else "",
            "annotation_exists": bool(annotation_dir and annotation_dir.exists()),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def preview_videos(case: dict):
    frame_idx = st.slider("Preview frame lokal", min_value=0, max_value=2000, value=0, step=1, key=f"preview_{case['case_id']}")
    grid = make_preview_grid(case["video_files"], frame_idx=frame_idx)
    if grid is not None:
        st.image(grid, channels="BGR", use_container_width=True)


def show_gallery(gallery_df: pd.DataFrame, title: str):
    st.subheader(title)
    if gallery_df is None or len(gallery_df) == 0:
        st.info("Gallery belum tersedia.")
        return
    cols = st.columns(4)
    for i, (_, row) in enumerate(gallery_df.iterrows()):
        path = Path(row["image_path"])
        with cols[i % 4]:
            if path.exists():
                st.image(str(path), use_container_width=True)
            label = row.get("global_id", row.get("track_key", ""))
            st.caption(str(label))


def download_table(df, label: str, filename: str, key: str | None = None):
    if df is None or len(df) == 0:
        return

    if key is None:
        safe_label = str(label).replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
        safe_filename = str(filename).replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
        key = f"download_{safe_label}_{safe_filename}_{id(df)}"

    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )
