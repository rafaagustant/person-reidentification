from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from core.video_io import make_preview_grid
try:
    from config.gt_cases import GT_CASE_META
except Exception:
    GT_CASE_META = {}


def widget_key(case_id: str, section: str, name: str, camera: str | None = None) -> str:
    parts = [case_id, section, name]
    if camera:
        parts.append(camera)
    return "__".join(str(part).replace(" ", "_") for part in parts)


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


def dataframe_tools(df: pd.DataFrame, filename: str, key: str) -> None:
    if df is None or len(df) == 0:
        return
    with st.expander("Salin atau unduh tabel"):
        csv_data = df.to_csv(index=False)
        tsv_data = df.to_csv(index=False, sep="\t")
        st.code(csv_data, language="csv")
        st.code(tsv_data, language="text")
        st.download_button("Unduh CSV", csv_data.encode("utf-8"), file_name=filename, mime="text/csv", key=f"{key}_csv")


def filtered_track_gallery(tracks_df: pd.DataFrame) -> None:
    """Render track yang ditolak oleh filter tanpa mengubah keputusannya."""
    columns = st.columns(4)
    for index, (_, row) in enumerate(tracks_df.head(20).iterrows()):
        with columns[index % 4]:
            st.markdown(f"**{row.get('track_key', 'Track')}**")
            image_path = Path(str(row.get("representative_crop", row.get("crop_path", ""))))
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.caption("Crop tidak tersedia")
            st.caption(
                f"{row.get('camera', '-')} | local track {row.get('track_id', '-')}\n\n"
                f"Frame: {row.get('num_frames', '-')} | Crop: {row.get('num_crops', '-')}\n\n"
                f"Avg conf: {row.get('avg_conf', '-')} | Avg area: {row.get('avg_area', '-')}"
            )
            st.caption(f"Alasan filter: {row.get('filter_reasons', '-')}")
            limits = []
            reasons = str(row.get("filter_reasons", ""))
            for value, label in [
                (row.get("min_frames_used"), "min_frames"),
                (row.get("min_crops_used"), "min_crops"),
                (row.get("min_avg_conf_used"), "min_avg_conf"),
                (row.get("min_avg_area_used"), "min_avg_area"),
            ]:
                if label in reasons and pd.notna(value):
                    limits.append(f"{label}: {value}")
            if limits:
                st.caption("Batas: " + " | ".join(limits))


def _render_merged_global_card(group: pd.Series, members: pd.DataFrame) -> None:
    with st.container(border=True):
        global_id = group["global_id"]
        st.subheader(f"Global ID {global_id}")
        st.caption(f"{group['num_member_tracks']} local track digabungkan | Kamera: {group['cameras']}")
        extra = []
        if pd.notna(group.get("dominant_gt_id")):
            extra.append(f"GT dominan: {group['dominant_gt_id']}")
        if pd.notna(group.get("global_id_purity")):
            extra.append(f"Purity: {float(group['global_id_purity']):.3f}")
        if extra:
            st.caption(" | ".join(extra))
        for start in range(0, len(members), 4):
            row_members = members.iloc[start:start + 4]
            columns = st.columns(len(row_members))
            for column, (_, member) in zip(columns, row_members.iterrows()):
                with column:
                    image_path = Path(str(member.get("crop_path", "")))
                    if image_path.exists():
                        st.image(str(image_path), width=200)
                    else:
                        st.caption("Crop tidak tersedia")
                    st.caption(
                        f"{member.get('track_key', '-')}\n\n"
                        f"{member.get('camera', '-')} | Track {member.get('local_track_id', '-')}"
                    )


def global_id_gallery(cards_df: pd.DataFrame) -> None:
    """Render one representative crop for every final Global ID."""
    if cards_df is None or len(cards_df) == 0:
        st.info("Jalankan Re-ID untuk menampilkan Global ID Gallery.")
        return
    cards = cards_df.copy()
    cards["_global_order"] = pd.to_numeric(cards["global_id"], errors="coerce")
    cards = cards.sort_values(["_global_order", "global_id"], kind="stable").drop(columns="_global_order")
    for start in range(0, len(cards), 3):
        row_cards = cards.iloc[start:start + 3]
        columns = st.columns(len(row_cards))
        for column, (_, card) in zip(columns, row_cards.iterrows()):
            with column:
                with st.container(border=True):
                    st.subheader(f"Global ID {card['global_id']}")
                    image_path = Path(str(card.get("representative_crop_path", "")))
                    if image_path.exists():
                        st.image(str(image_path), width=200)
                    else:
                        st.caption("Crop representatif tidak tersedia.")
                    st.caption(
                        f"Crop representatif: {card.get('representative_track_key', '-')}\n\n"
                        f"{card.get('representative_camera', '-')} | Local Track ID {card.get('representative_local_track_id', '-')}"
                    )
                    st.caption(f"{card['num_member_tracks']} local track | {card['num_cameras']} kamera")
                    st.markdown("Dibentuk dari Local Track")
                    st.caption(card["member_track_keys"])
                    st.caption(f"Kamera: {card['cameras']}")
                    extra = []
                    if pd.notna(card.get("dominant_gt_id")):
                        extra.append(f"GT Dominan: {card['dominant_gt_id']}")
                    if pd.notna(card.get("global_id_purity")):
                        extra.append(f"Purity: {float(card['global_id_purity']):.3f}")
                    if extra:
                        st.caption(" | ".join(extra))


def merged_track_gallery(groups_df: pd.DataFrame, members_df: pd.DataFrame) -> None:
    """Render each final multi-member Global ID as one gallery card."""
    groups = groups_df.copy()
    groups["_global_order"] = pd.to_numeric(groups["global_id"], errors="coerce")
    groups = groups.sort_values(["_global_order", "global_id"], kind="stable").drop(columns="_global_order")
    index = 0
    while index < len(groups):
        group = groups.iloc[index]
        group_members = members_df[members_df["global_id"] == group["global_id"]]
        if int(group["num_member_tracks"]) > 4:
            _render_merged_global_card(group, group_members)
            index += 1
            continue
        columns = st.columns(2)
        for column in columns:
            if index >= len(groups):
                break
            current = groups.iloc[index]
            current_members = members_df[members_df["global_id"] == current["global_id"]]
            with column:
                _render_merged_global_card(current, current_members)
            index += 1


def crop_gallery(gallery_df: pd.DataFrame, title: str, key: str) -> None:
    st.subheader(title)
    if gallery_df is None or len(gallery_df) == 0:
        st.info("Data kosong.")
        return
    columns = st.columns(4)
    for index, (_, row) in enumerate(gallery_df.iterrows()):
        with columns[index % 4]:
            image_path = Path(str(row.get("image_path", row.get("representative_crop", ""))))
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.caption("Crop tidak tersedia")
            labels = [str(row.get(name)) for name in ["global_id", "track_key", "camera", "track_id"] if name in row and pd.notna(row.get(name))]
            st.caption(" · ".join(labels))
