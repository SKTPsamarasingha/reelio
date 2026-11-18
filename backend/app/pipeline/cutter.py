"""Cut one clip with ffmpeg: crop or fit to the chosen layout, optionally burning in captions."""
import logging
import os
import subprocess
from functools import lru_cache

from app.options import LAYOUTS
from app.pipeline.media import ffmpeg_exe

log = logging.getLogger(__name__)


@lru_cache
def captions_supported() -> bool:
    """True if this ffmpeg build has the 'ass' filter (libass), which burns captions in."""
    try:
        out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=20).stdout
        return any(line.split()[1:2] == ["ass"] for line in out.splitlines())
    except Exception:
        return False


def _esc(path: str) -> str:
    """Escape a file path for use inside an ffmpeg filter (Windows drive colons included)."""
    return str(path).replace("\\", "/").replace(":", r"\:")


def _crop_filter(w, h, focus_x, focus_y):
    # Largest w:h region, never stretched. focus: 0 = left/top, 0.5 = centre, 1 = right/bottom.
    return (
        f"crop=w='min(iw,ih*{w}/{h})':h='min(ih,iw*{h}/{w})':"
        f"x='(iw-ow)*{focus_x}':y='(ih-oh)*{focus_y}',"
        f"scale={w}:{h}:flags=lanczos,setsar=1"
    )


def _piecewise(keys):
    """Linear interpolation between (t, value) keyframes as an ffmpeg expression in t."""
    expr = f"{keys[-1][1]:.4f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(keys, keys[1:]))):
        if t1 - t0 < 1e-6:
            continue
        expr = f"if(lt(t,{t1:.3f}),{x0:.4f}+({x1 - x0:.4f})*(t-{t0:.3f})/{t1 - t0:.3f},{expr})"
    return expr


def _follow_filter(w, h, track, focus_y):
    crop_frac = min(1.0, (track.src_h * w / h) / track.src_w)
    keys = track.keyframes(crop_frac)
    if not keys:
        return _crop_filter(w, h, 0.5, focus_y)
    return (
        f"crop=w='min(iw,ih*{w}/{h})':h='min(ih,iw*{h}/{w})':"
        f"x='(iw-ow)*({_piecewise(keys)})':y='(ih-oh)*{focus_y}',"
        f"scale={w}:{h}:flags=lanczos,setsar=1"
    )


def _fit_filter(w, h):
    # Whole frame fitted inside, with a blurred zoomed copy filling the background.
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def cut_video_clip(
    video_path,
    start_time,
    end_time,
    output_dir,
    output_filename,
    layout="9:16",          # key of options.LAYOUTS
    framing="fill",         # "fill" = crop (follows faces when a track is given), "fit" = whole frame on a blurred background
    track=None,             # FaceTrack from reframe.track_faces
    subtitles=None,         # path to an .ass file to burn in
    fonts_dir=None,
    focus_y=0.5,
):
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Options: {list(LAYOUTS)}")
    w, h = LAYOUTS[layout].width, LAYOUTS[layout].height

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"Invalid clip range: {start_time} to {end_time}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    burn = ""
    if subtitles:
        burn = f",ass=filename='{_esc(subtitles)}'" + (f":fontsdir='{_esc(fonts_dir)}'" if fonts_dir else "")

    cmd = [ffmpeg_exe(), "-y", "-ss", str(start_time), "-i", str(video_path), "-t", str(duration)]
    if framing == "fit":
        cmd += ["-filter_complex", f"[0:v]{_fit_filter(w, h)}{burn}[v]", "-map", "[v]", "-map", "0:a?"]
    elif track is not None:
        cmd += ["-vf", _follow_filter(w, h, track, focus_y) + burn]
    else:
        cmd += ["-vf", _crop_filter(w, h, 0.5, focus_y) + burn]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr[-1500:])      # details go to the log, not to the user
        raise RuntimeError("FFmpeg failed to cut the clip.")
    return output_path
