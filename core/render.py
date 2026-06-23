from __future__ import annotations

from pathlib import Path
import math
import subprocess
import cv2
import numpy as np
import pandas as pd

from utils.helpers import ensure_dir


BRIGHT_COLORS = [
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (0, 180, 255),    # orange
    (80, 255, 80),    # bright green
    (255, 120, 80),   # light blue
    (120, 80, 255),   # pink/purple
    (0, 255, 140),    # lime
    (255, 200, 0),    # sky-ish
    (180, 255, 0),    # aqua-green
    (0, 128, 255),    # amber
    (255, 0, 128),    # violet
]


def color_from_id(identity_id: int):
    identity_id = int(identity_id)
    if 1 <= identity_id <= len(BRIGHT_COLORS):
        return BRIGHT_COLORS[identity_id - 1]

    hue = (identity_id * 67) % 180
    hsv = np.array([[[hue, 230, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def text_color_for_bg(color):
    b, g, r = [int(x) for x in color]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance >= 150 else (255, 255, 255)


def draw_transparent_rect(frame, x1, y1, x2, y2, color, alpha=0.65) -> None:
    x1 = max(0, min(frame.shape[1] - 1, int(x1)))
    x2 = max(0, min(frame.shape[1] - 1, int(x2)))
    y1 = max(0, min(frame.shape[0] - 1, int(y1)))
    y2 = max(0, min(frame.shape[0] - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        overlay[y1:y2, x1:x2],
        float(alpha),
        frame[y1:y2, x1:x2],
        1.0 - float(alpha),
        0,
    )


def _rect_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _clamp_rect(rect, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    x1 = max(0, min(frame_w - width - 1, x1))
    y1 = max(0, min(frame_h - height - 1, y1))
    x2 = min(frame_w - 1, x1 + width)
    y2 = min(frame_h - 1, y1 + height)
    return int(x1), int(y1), int(x2), int(y2)


def _place_label_rect(candidates, used_label_rects, frame_w: int, frame_h: int):
    best_rect = None
    for rect in candidates:
        rect = _clamp_rect(rect, frame_w, frame_h)
        best_rect = rect
        for shift in [0, 8, 16, 24, 32]:
            shifted = _clamp_rect(
                (rect[0], rect[1] + shift, rect[2], rect[3] + shift),
                frame_w,
                frame_h,
            )
            if not any(_rect_overlap(shifted, old) for old in used_label_rects):
                return shifted
            best_rect = shifted
    return best_rect


def _draw_labeled_box(frame, box, label: str, color, used_label_rects=None):
    if used_label_rects is None:
        used_label_rects = []

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(0, min(frame_w - 1, x2))
    y1 = max(0, min(frame_h - 1, y1))
    y2 = max(0, min(frame_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return used_label_rects

    box_w = x2 - x1
    box_h = y2 - y1
    thickness = 1 if box_w < 70 or box_h < 100 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    compact = box_w < 80 or box_h < 100
    font_scale = 0.40 if compact else 0.50
    text_thickness = 1
    pad_x = 4 if compact else 5
    pad_y = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    label_w = text_w + (2 * pad_x)
    label_h = text_h + baseline + (2 * pad_y)
    top_rect = (x1, y1 - label_h - 3, x1 + label_w, y1 - 3)
    bottom_rect = (x1, y2 + 3, x1 + label_w, y2 + label_h + 3)
    inside_rect = (x1 + 1, y1 + 1, x1 + label_w + 1, y1 + label_h + 1)
    candidates = [top_rect, bottom_rect, inside_rect]

    label_rect = _place_label_rect(candidates, used_label_rects, frame_w, frame_h)
    if label_rect is None:
        return used_label_rects

    label_x1, label_y1, label_x2, label_y2 = label_rect
    draw_transparent_rect(frame, label_x1, label_y1, label_x2, label_y2, color, alpha=0.65)
    text_color = text_color_for_bg(color)
    text_org = (label_x1 + pad_x, label_y2 - baseline - pad_y)
    cv2.putText(frame, label, text_org, font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    used_label_rects.append(label_rect)
    return used_label_rects


def render_camera_video(
    video_path: str | Path,
    camera: str,
    tracks_df: pd.DataFrame,
    output_path: str | Path,
    label_mode: str = "global",
    show_gt: bool = False,
    progress_callback=None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cam_df = tracks_df[tracks_df["camera"] == camera].copy() if tracks_df is not None and len(tracks_df) else pd.DataFrame()
    frame_map = {int(k): v for k, v in cam_df.groupby("frame")} if len(cam_df) else {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        used_label_rects = []
        rows = frame_map.get(frame_idx)
        if rows is not None and len(rows):
            for _, r in rows.iterrows():
                x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
                tid = int(r["track_id"])
                gid = int(r["global_id"]) if "global_id" in r and pd.notna(r["global_id"]) else tid
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                if label_mode == "global":
                    color = color_from_id(gid)
                    if box_w < 80 or box_h < 100:
                        label = f"G{gid}|T{tid}"
                    else:
                        label = f"GID:{gid} | T{tid}"
                else:
                    color = color_from_id(tid)
                    label = f"T{tid}"
                if show_gt and "gt_id" in r:
                    label += f" | GT{int(r['gt_id'])}"
                used_label_rects = _draw_labeled_box(
                    frame,
                    (x1, y1, x2, y2),
                    label,
                    color,
                    used_label_rects=used_label_rects,
                )
        cv2.putText(frame, f"{camera} | frame {frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
        if progress_callback is not None:
            progress_callback(camera, frame_idx, total)
    cap.release()
    writer.release()
    return output_path


def _resize_to_cell(frame, cell_w, cell_h, bg_color=(8, 10, 14)):
    h, w = frame.shape[:2]
    scale = min(cell_w / max(1, w), cell_h / max(1, h))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((cell_h, cell_w, 3), bg_color, dtype=np.uint8)
    x0 = (cell_w - nw) // 2
    y0 = (cell_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _fullscreen_layout_spec(n: int, out_w: int, out_h: int, padding: int):
    if n <= 1:
        return [(0, 0, out_w, out_h)]
    if n == 2:
        tile_w = (out_w - padding * 3) // 2
        tile_h = out_h - padding * 2
        return [
            (padding, padding, tile_w, tile_h),
            (padding * 2 + tile_w, padding, tile_w, tile_h),
        ]
    main_w = int(out_w * 0.66) - padding
    side_w = out_w - main_w - padding * 3
    main_h = out_h - padding * 2
    side_h = (out_h - padding * 3) // 2
    return [
        (padding, padding, main_w, main_h),
        (padding * 2 + main_w, padding, side_w, side_h),
        (padding * 2 + main_w, padding * 2 + side_h, side_w, side_h),
    ]


def combine_videos_grid(
    video_paths: list[str | Path],
    output_path: str | Path,
    cell_w=640,
    cell_h=360,
    layout: str = "fullscreen_grid",
    output_width: int | None = None,
    output_height: int | None = None,
    padding: int = 6,
    info_panel: dict | None = None,
) -> Path | None:
    video_paths = [Path(p) for p in video_paths if Path(p).exists()]
    if not video_paths:
        return None
    if len(video_paths) > 3:
        video_paths = video_paths[:3]
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25
    n = len(caps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if layout == "fullscreen_grid":
        out_w = int(output_width or 1920)
        out_h = int(output_height or 1080)
        specs = _fullscreen_layout_spec(n, out_w, out_h, padding)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        while True:
            frames = []
            any_ok = False
            for cap in caps:
                ok, frame = cap.read()
                if ok:
                    any_ok = True
                    frames.append(frame)
                else:
                    frames.append(None)
            if not any_ok:
                break
            canvas = np.full((out_h, out_w, 3), (8, 10, 14), dtype=np.uint8)
            for idx, spec in enumerate(specs[:n]):
                x, y, w, h = spec
                frame = frames[idx]
                tile = _resize_to_cell(frame, w, h) if frame is not None else np.full((h, w, 3), (8, 10, 14), dtype=np.uint8)
                canvas[y:y+h, x:x+w] = tile
            writer.write(canvas)
        for cap in caps:
            cap.release()
        writer.release()
        return output_path

    if n == 1:
        cols = 1
        rows = 1
        padding = 0
    elif layout in {"auto_fit", "horizontal"}:
        cols = n
        rows = 1
    else:
        cols = min(2, n)
        rows = math.ceil(n / cols)

    if output_width is None:
        output_width = 1280 if n == 2 else 1440 if n == 3 else cell_w
    if n > 1 and layout in {"auto_fit", "horizontal"}:
        cell_w = max(1, int((int(output_width) - padding * (cols - 1)) / cols))
        cell_h = int(round(cell_w * 9 / 16))

    out_w = cols * cell_w + padding * max(0, cols - 1)
    out_h = rows * cell_h + padding * max(0, rows - 1)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    while True:
        frames = []
        any_ok = False
        for cap in caps:
            ok, frame = cap.read()
            if ok:
                any_ok = True
                frames.append(_resize_to_cell(frame, cell_w, cell_h))
            else:
                frames.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        if not any_ok:
            break
        while len(frames) < rows * cols:
            frames.append(np.full((cell_h, cell_w, 3), (8, 10, 14), dtype=np.uint8))
        grid_rows = []
        for r in range(rows):
            row_frames = frames[r*cols:(r+1)*cols]
            if padding and len(row_frames) > 1:
                sep = np.full((cell_h, padding, 3), (8, 10, 14), dtype=np.uint8)
                row = row_frames[0]
                for frame in row_frames[1:]:
                    row = np.hstack([row, sep, frame])
                grid_rows.append(row)
            else:
                grid_rows.append(np.hstack(row_frames))
        if padding and len(grid_rows) > 1:
            sep = np.full((padding, out_w, 3), (8, 10, 14), dtype=np.uint8)
            grid = grid_rows[0]
            for row in grid_rows[1:]:
                grid = np.vstack([grid, sep, row])
        else:
            grid = np.vstack(grid_rows)
        writer.write(grid)
    for cap in caps:
        cap.release()
    writer.release()
    return output_path


def make_streamlit_playable(video_path: str | Path) -> Path:
    video_path = Path(video_path)
    out = video_path.with_name(video_path.stem + "_h264.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out if out.exists() else video_path
    except Exception:
        return video_path
