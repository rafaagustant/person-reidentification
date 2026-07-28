from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _source_frame_range(row: pd.Series) -> str:
    first, last = row.get("first_source_frame"), row.get("last_source_frame")
    if pd.isna(first) or pd.isna(last):
        return "Tidak tersedia"
    return str(int(first)) if int(first) == int(last) else f"{int(first)}–{int(last)}"


def _ratio_label(value) -> str:
    return "Tidak tersedia" if value is None or pd.isna(value) else f"{float(value):.0%}"


def _decimal_label(value) -> str:
    return "Tidak tersedia" if value is None or pd.isna(value) else f"{float(value):.3f}"


def _text_label(value, fallback: str = "-") -> str:
    return fallback if value is None or pd.isna(value) or str(value).strip() == "" else str(value)


def _main_unmatched_reason(row: pd.Series) -> str:
    reasons = [
        ("gt_claimed_by_other_count", "GT sudah dipasangkan dengan track lain"),
        ("below_iou_threshold_count", "IoU terbaik berada di bawah threshold"),
        ("no_gt_on_frame_count", "Tidak ada GT pada frame"),
        ("unknown_unmatched_count", "Penyebab tidak dapat diklasifikasikan"),
    ]
    counts = [
        (0 if pd.isna(row.get(column, 0)) else int(row.get(column, 0)), label)
        for column, label in reasons
    ]
    count, label = max(counts, default=(0, ""))
    return f"Alasan utama: {label} pada {count} prediksi." if count else "Alasan utama: tidak ada prediksi unmatched."


def widget_key(case_id: str, section: str, name: str, camera: str | None = None) -> str:
    parts = [case_id, section, name]
    if camera:
        parts.append(camera)
    return "__".join(str(part).replace(" ", "_") for part in parts)


def dataframe_tools(df: pd.DataFrame, filename: str, key: str) -> None:
    if df is None or len(df) == 0:
        return
    with st.expander("Salin atau unduh tabel"):
        csv_data = df.to_csv(index=False)
        tsv_data = df.to_csv(index=False, sep="\t")
        st.code(csv_data, language="csv")
        st.code(tsv_data, language="text")
        st.download_button("Unduh CSV", csv_data.encode("utf-8"), file_name=filename, mime="text/csv", key=f"{key}_csv")


def dataframe_with_tools(
    df: pd.DataFrame,
    filename: str,
    key: str,
    *,
    hide_index: bool = True,
) -> None:
    """Tampilkan tabel beserta CSV dan TSV yang dapat disalin."""
    st.dataframe(df, hide_index=hide_index, key=f"{key}__table")
    dataframe_tools(df, filename, key)


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
                f"Frame: {row.get('num_frames', '-')} | Rentang: {_source_frame_range(row)} | Continuity: {_ratio_label(row.get('continuity_ratio'))}\n\n"
                f"Crop: {row.get('num_crops', '-')}\n\n"
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
            track_key = str(row.get("track_key", "Track"))
            st.markdown(f"**{track_key}**")
            st.caption(
                f"Frame: {row.get('num_frames', '-')} | Rentang: {_source_frame_range(row)} | Continuity: {_ratio_label(row.get('continuity_ratio'))}\n\n"
                f"Avg conf: {row.get('avg_conf', '-')} | Avg area: {row.get('avg_area', '-')}\n\n"
                f"Status: {row.get('status', 'Valid')}"
            )
            st.caption(f"Alasan filter: {_text_label(row.get('filter_reason'))}")
            if _truthy(row.get("evaluation_included")):
                st.caption(f"Evaluasi GT\n\nMatched: {row.get('matched_frame_count', 0)} | Unmatched: {row.get('unmatched_frame_count', 0)} | Match rate: {_ratio_label(row.get('match_rate'))}")
                st.caption(_main_unmatched_reason(row))
                with st.expander(f"Detail Diagnosis GT — {track_key}"):
                    st.caption(
                        f"no_gt_on_frame: {row.get('no_gt_on_frame_count', 0)}\n\n"
                        f"below_iou_threshold: {row.get('below_iou_threshold_count', 0)}\n\n"
                        f"gt_claimed_by_other_prediction: {row.get('gt_claimed_by_other_count', 0)}\n\n"
                        f"unknown: {row.get('unknown_unmatched_count', 0)}\n\n"
                        f"Candidate GT ID: {row.get('candidate_gt_id', '-')}\n\n"
                        f"Best IoU mean/max: {_decimal_label(row.get('best_iou_mean'))} / {_decimal_label(row.get('best_iou_max'))}\n\n"
                        f"GT diklaim track: {row.get('claimed_by_track_key', '-')} ({row.get('claimed_by_frame_count', 0)} frame)\n\n"
                        f"IoU threshold: {_decimal_label(row.get('iou_threshold_used'))}"
                    )
            labels = [str(row.get(name)) for name in ["global_id", "track_key", "camera", "track_id"] if name in row and pd.notna(row.get(name))]
            st.caption(" · ".join(labels))
