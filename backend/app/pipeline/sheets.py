"""Contact sheets: one frame per transcript segment in 2x2 grids, each tile stamped with its segment id and time."""
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from app.pipeline.media import open_video, probe_video, read_frame_at
from app.pipeline.progress import Bar, ok

GRID_COLS = 2
MAX_TILES_PER_SHEET = 4     # 2x2
TILE_WIDTH = 512
JPEG_QUALITY = 80


@dataclass
class Sheet:
    image_id: str
    path: Path
    start: float
    end: float
    grid: str
    frames: list[dict]      # {segment_id, timestamp, tile_position}


def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def split_evenly(n_items: int, max_per_sheet: int = MAX_TILES_PER_SHEET) -> list[tuple[int, int]]:
    """[(first_index, size)] spreading items over the fewest sheets, so the last sheet is never a tiny leftover."""
    if n_items == 0:
        return []
    n_sheets = math.ceil(n_items / max_per_sheet)
    base, extra = divmod(n_items, n_sheets)
    groups, start = [], 0
    for i in range(n_sheets):
        size = base + (1 if i < extra else 0)
        groups.append((start, size))
        start += size
    return groups


def _make_tile(frame, label, tile_w, tile_h):
    import cv2
    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(tile, (0, 0), (text_w + 14, text_h + baseline + 12), (0, 0, 0), -1)
    cv2.putText(tile, label, (7, text_h + 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return tile


def _stitch(tiles, tile_w, tile_h):
    rows = math.ceil(len(tiles) / GRID_COLS)
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    padded = tiles + [blank] * (rows * GRID_COLS - len(tiles))
    grid = np.vstack([np.hstack(padded[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(rows)])
    return grid, rows


def build_contact_sheets(video_path, segments, out_dir) -> list[Sheet]:
    """Write the sheet JPEGs into out_dir and describe them. Nothing is uploaded; they only feed the model."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    tile_w = TILE_WIDTH
    tile_h = max(int(tile_w * info.height / info.width) // 2 * 2, 2)

    bar = Bar("contact sheets", len(segments) or 1)
    sheets, done = [], 0
    with open_video(video_path) as cap:
        for sheet_no, (first, size) in enumerate(split_evenly(len(segments)), start=1):
            tiles, frames = [], []
            for pos, seg in enumerate(segments[first:first + size]):
                seg_no = first + pos
                mid = (seg["start"] + seg["end"]) / 2
                frame = read_frame_at(cap, mid)
                tiles.append(
                    _make_tile(frame, f"seg_{seg_no} | {fmt_time(mid)}", tile_w, tile_h)
                    if frame is not None else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                )
                frames.append({"segment_id": f"seg_{seg_no}", "timestamp": round(mid, 2), "tile_position": pos})
                done += 1
                bar.update(done)

            grid, rows = _stitch(tiles, tile_w, tile_h)
            image_id = f"sheet_{sheet_no:03d}"
            path = out_dir / f"{image_id}.jpg"
            if not cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise RuntimeError(f"Could not save contact sheet {path}")
            sheets.append(Sheet(image_id, path, frames[0]["timestamp"], frames[-1]["timestamp"],
                                f"{GRID_COLS}x{rows}", frames))
    bar.finish()
    ok(f"{len(sheets)} contact sheet(s) built")
    return sheets
