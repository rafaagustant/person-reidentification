from __future__ import annotations

import itertools
import json
import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # A stable root keeps Global ID assignment independent of row order.
            self.parent[max(ra, rb)] = min(ra, rb)


def _temporal_gap_overlap(a: dict, b: dict) -> tuple[int, int]:
    start_a, end_a = int(a["first_frame"]), int(a["last_frame"])
    start_b, end_b = int(b["first_frame"]), int(b["last_frame"])
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    if overlap > 0:
        gap = 0
    elif end_a < start_b:
        gap = start_b - end_a - 1
    elif end_b < start_a:
        gap = start_a - end_b - 1
    else:
        gap = 0
    return int(gap), int(overlap)


def _source_frame_set(track: pd.Series | dict) -> set[int] | None:
    value = track.get("source_frames")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return None
    return {int(frame) for frame in value if pd.notna(frame)}


def _actual_temporal_gap_overlap(a: pd.Series | dict, b: pd.Series | dict) -> tuple[int, int]:
    frames_a = _source_frame_set(a)
    frames_b = _source_frame_set(b)
    if not frames_a or not frames_b:
        return _temporal_gap_overlap(a, b)

    overlap = len(frames_a.intersection(frames_b))
    if overlap:
        return 0, int(overlap)

    start_a, end_a = min(frames_a), max(frames_a)
    start_b, end_b = min(frames_b), max(frames_b)
    if end_a < start_b:
        return int(start_b - end_a - 1), 0
    if end_b < start_a:
        return int(start_a - end_b - 1), 0
    return 0, 0


def _source_temporal_metrics(a: pd.Series, b: pd.Series) -> dict:
    frames_a = _source_frame_set(a)
    frames_b = _source_frame_set(b)
    fallback = {
        "start_source_frame_a": a.get("first_source_frame", a.get("first_frame")),
        "end_source_frame_a": a.get("last_source_frame", a.get("last_frame")),
        "start_source_frame_b": b.get("first_source_frame", b.get("first_frame")),
        "end_source_frame_b": b.get("last_source_frame", b.get("last_frame")),
        "overlap_frame_count": pd.NA,
        "temporal_gap": pd.NA,
        "temporal_status": "Tidak tersedia",
    }
    if not frames_a or not frames_b:
        return fallback
    start_a, end_a = min(frames_a), max(frames_a)
    start_b, end_b = min(frames_b), max(frames_b)
    overlap = len(frames_a.intersection(frames_b))
    if overlap:
        gap, status = 0, "Overlap"
    elif end_a < start_b:
        gap, status = start_b - end_a - 1, "Berurutan" if start_b - end_a - 1 == 0 else "Terpisah"
    elif end_b < start_a:
        gap, status = start_a - end_b - 1, "Berurutan" if start_a - end_b - 1 == 0 else "Terpisah"
    else:
        gap, status = 0, "Berurutan"
    return {
        "start_source_frame_a": start_a,
        "end_source_frame_a": end_a,
        "start_source_frame_b": start_b,
        "end_source_frame_b": end_b,
        "overlap_frame_count": overlap,
        "temporal_gap": int(max(0, gap)),
        "temporal_status": status,
    }


def _merged_cluster_members(uf: UnionFind, track_keys: list[str], a: str, b: str) -> list[str]:
    roots = {uf.find(a), uf.find(b)}
    return [tk for tk in track_keys if uf.find(tk) in roots]


def _cluster_temporal_conflict(
    members: list[str],
    track_meta: dict[str, dict],
    candidate: tuple[str, str],
    intra_max_gap: int,
    intra_max_overlap: int,
) -> tuple[bool, str]:
    candidate_set = set(candidate)
    for left, right in itertools.combinations(members, 2):
        left_meta = track_meta[left]
        right_meta = track_meta[right]
        if left_meta["camera"] != right_meta["camera"]:
            continue

        gap, overlap = _actual_temporal_gap_overlap(left_meta, right_meta)
        if overlap > intra_max_overlap or gap > intra_max_gap:
            kind = "overlap" if overlap > intra_max_overlap else "gap"
            scope = "same_camera" if {left, right} == candidate_set else "cluster"
            return True, f"blocked_{scope}_temporal_{kind}"

    return False, ""


def compute_track_similarity_df(track_df: pd.DataFrame, track_features: np.ndarray) -> pd.DataFrame:
    rows = []
    if track_df is None or len(track_df) <= 1 or track_features.size == 0:
        return pd.DataFrame(columns=[
            "track_a",
            "camera_a",
            "first_frame_a",
            "last_frame_a",
            "track_b",
            "camera_b",
            "first_frame_b",
            "last_frame_b",
            "same_camera",
            "association_temporal_gap",
            "association_temporal_overlap",
            "temporal_gap",
            "temporal_overlap",
            "start_source_frame_a",
            "end_source_frame_a",
            "start_source_frame_b",
            "end_source_frame_b",
            "overlap_frame_count",
            "temporal_status",
            "cosine_similarity",
        ])

    sim = track_features @ track_features.T
    for i, j in itertools.combinations(range(len(track_df)), 2):
        a = track_df.iloc[i]
        b = track_df.iloc[j]
        gap, overlap = _temporal_gap_overlap(a, b)
        source_temporal = _source_temporal_metrics(a, b)
        rows.append({
            "track_a": a["track_key"],
            "camera_a": a["camera"],
            "first_frame_a": int(a["first_frame"]),
            "last_frame_a": int(a["last_frame"]),
            "track_b": b["track_key"],
            "camera_b": b["camera"],
            "first_frame_b": int(b["first_frame"]),
            "last_frame_b": int(b["last_frame"]),
            "same_camera": a["camera"] == b["camera"],
            "association_temporal_gap": int(gap),
            "association_temporal_overlap": int(overlap),
            "temporal_overlap": int(overlap),
            "cosine_similarity": float(sim[i, j]),
            **source_temporal,
        })
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)


def mutual_nearest_cross_pairs(pair_df: pd.DataFrame, threshold: float) -> set[tuple[str, str]]:
    cross = pair_df[(~pair_df["same_camera"]) & (pair_df["cosine_similarity"] >= threshold)].copy()
    if len(cross) == 0:
        return set()

    pairs = set()
    # MNN is evaluated independently for every camera pair. A track may therefore
    # form valid transitive links with tracks from more than one other camera.
    camera_pairs = cross.apply(
        lambda row: tuple(sorted((str(row["camera_a"]), str(row["camera_b"])))), axis=1
    )
    for camera_pair in sorted(camera_pairs.unique()):
        pair_rows = cross[camera_pairs == camera_pair]
        directed = []
        for _, row in pair_rows.iterrows():
            directed.extend((
                {"src": str(row["track_a"]), "dst": str(row["track_b"]), "sim": float(row["cosine_similarity"])},
                {"src": str(row["track_b"]), "dst": str(row["track_a"]), "sim": float(row["cosine_similarity"])},
            ))
        directed_df = pd.DataFrame(directed)
        best = directed_df.sort_values(["src", "sim", "dst"], ascending=[True, False, True]).drop_duplicates("src")
        best_map = dict(zip(best["src"], best["dst"]))
        for a, b in best_map.items():
            if best_map.get(b) == a:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def assign_global_ids(track_df: pd.DataFrame, pair_df: pd.DataFrame, reid_cfg: dict):
    if track_df is None or len(track_df) == 0:
        return pd.DataFrame(), pair_df

    track_keys = track_df["track_key"].tolist()
    track_meta = {
        row["track_key"]: {
            "camera": row["camera"],
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "source_frames": row.get("source_frames"),
        }
        for _, row in track_df.iterrows()
    }
    uf = UnionFind(track_keys)
    pair_df = pair_df.copy()
    pair_df["merge_status"] = False
    pair_df["merge_reason"] = "not_merged"

    cross_threshold = float(reid_cfg.get("cross_threshold", 0.75))
    intra_threshold = float(reid_cfg.get("intra_threshold", 0.80))
    intra_max_gap = int(reid_cfg.get("intra_max_gap", 30))
    intra_max_overlap = int(reid_cfg.get("intra_max_overlap", 0))
    pair_df["association_type"] = pair_df["same_camera"].map(lambda value: "intra_camera" if bool(value) else "cross_camera")
    pair_df["threshold"] = pair_df["same_camera"].map(lambda value: intra_threshold if bool(value) else cross_threshold)
    pair_df["similarity_margin"] = pair_df["cosine_similarity"].astype(float) - pair_df["threshold"].astype(float)
    pair_df["rejection_reason"] = ""

    mnn_pairs = set()
    if reid_cfg.get("use_mnn", True):
        mnn_pairs = mutual_nearest_cross_pairs(pair_df, cross_threshold)

    def mnn_status_code(row: pd.Series) -> str:
        if bool(row["same_camera"]) or not reid_cfg.get("enable_cross_camera", False) or not reid_cfg.get("use_mnn", True):
            return "not_applicable"
        if float(row["cosine_similarity"]) < cross_threshold:
            return "not_evaluated"
        return "passed" if tuple(sorted((str(row["track_a"]), str(row["track_b"])))) in mnn_pairs else "failed"

    pair_df["mnn_status_code"] = pair_df.apply(mnn_status_code, axis=1)

    pair_df = pair_df.sort_values(
        ["cosine_similarity", "track_a", "track_b"], ascending=[False, True, True], kind="stable"
    )
    for idx, r in pair_df.iterrows():
        a, b = r["track_a"], r["track_b"]
        sim = float(r["cosine_similarity"])
        same_camera = bool(r["same_camera"])
        source_gap = r.get("temporal_gap")
        source_overlap = r.get("overlap_frame_count")
        gap = (
            int(source_gap)
            if pd.notna(source_gap)
            else int(r.get("association_temporal_gap", r["temporal_gap"]))
        )
        overlap = (
            int(source_overlap)
            if pd.notna(source_overlap)
            else int(r.get("association_temporal_overlap", r["temporal_overlap"]))
        )
        should_merge = False
        reason = "not_merged"
        rejection_reasons = []

        if not np.isfinite(sim):
            rejection_reasons.append("rejected_invalid_embedding")
        elif same_camera and reid_cfg.get("enable_strict_intra", False):
            if sim < intra_threshold:
                rejection_reasons.append("rejected_similarity")
            if gap > intra_max_gap:
                rejection_reasons.append("rejected_temporal_gap")
            if overlap > intra_max_overlap:
                rejection_reasons.append("rejected_temporal_overlap")
            if not rejection_reasons:
                should_merge = True
                reason = "intra_fragment_recovery"
        elif (not same_camera) and reid_cfg.get("enable_cross_camera", False):
            if sim < cross_threshold:
                rejection_reasons.append("rejected_similarity")
            else:
                if reid_cfg.get("use_mnn", True):
                    if tuple(sorted([a, b])) in mnn_pairs:
                        should_merge = True
                        reason = "cross_camera_mnn"
                    else:
                        rejection_reasons.append("rejected_mnn")
                else:
                    should_merge = True
                    reason = "cross_camera_threshold"
        else:
            rejection_reasons.append("rejected_camera_rule")

        if should_merge:
            proposed_members = _merged_cluster_members(uf, track_keys, a, b)
            has_conflict, conflict_reason = _cluster_temporal_conflict(
                proposed_members,
                track_meta,
                (a, b),
                intra_max_gap=intra_max_gap,
                intra_max_overlap=intra_max_overlap,
            )
            if has_conflict:
                should_merge = False
                reason = conflict_reason
                rejection_reasons.append(
                    "rejected_temporal_overlap" if "overlap" in conflict_reason else "rejected_temporal_gap"
                )
            else:
                uf.union(a, b)
                pair_df.at[idx, "merge_status"] = True
        pair_df.at[idx, "merge_reason"] = reason
        if not should_merge:
            pair_df.at[idx, "rejection_reason"] = ";".join(dict.fromkeys(rejection_reasons)) or "rejected_camera_rule"

    comps = {}
    for tk in track_keys:
        comps.setdefault(uf.find(tk), []).append(tk)

    rows = []
    ordered_components = sorted((sorted(members) for members in comps.values()), key=lambda members: members[0])
    for gid_num, members in enumerate(ordered_components, start=1):
        gid = gid_num
        for tk in members:
            rows.append({"track_key": tk, "global_id": gid})

    mapping = pd.DataFrame(rows)
    meta = track_df.merge(mapping, on="track_key", how="left")
    meta = meta.sort_values(["global_id", "camera", "track_key"]).reset_index(drop=True)
    return meta, pair_df
