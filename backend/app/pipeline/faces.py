"""Facial expressions per transcript segment, from MediaPipe's face blendshapes.

The labels (happy, surprise, anger...) come from simple rules on the blendshape scores.
They are a heuristic, not a trained emotion model. Also tells apart the face that is probably
talking (the one whose mouth moves) from listeners, so reaction shots can be found.
"""
import urllib.request
from collections import Counter
from dataclasses import dataclass

from app.config import BACKEND_DIR
from app.pipeline.media import open_video, read_frame_at, shrink
from app.pipeline.progress import Bar, ok

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = BACKEND_DIR / "models" / "face_landmarker.task"

MAX_FACES = 4
DETECT_MAX_SIDE = 720
MIN_FACE_AREA = 0.01       # ignore faces smaller than 1% of the frame (blendshapes are noise)
EMOTION_THRESHOLD = 0.35   # below this the face counts as neutral
TRACK_MATCH = 0.15         # faces closer than this (fraction of frame width) are the same person
MIN_JAW_MOVEMENT = 0.06    # mouth movement needed to call a face "the speaker"


@dataclass(frozen=True)
class Face:
    cx: float        # horizontal centre, 0-1
    width: float     # 0-1
    area: float      # fraction of the frame
    scores: dict     # blendshape name -> score


# ---------- model ----------
def ensure_model() -> str:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        bar = Bar("faces: downloading model", None)

        def hook(block, size, total):
            if total > 0:
                bar.update(min(block * size, total), total)

        part = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, part, reporthook=hook)
        part.replace(MODEL_PATH)
        bar.finish()
    return str(MODEL_PATH)


_landmarker = None


def get_landmarker():
    """One landmarker per worker process, shared with reframe.py."""
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=MAX_FACES,
            output_face_blendshapes=True,
        ))
    return _landmarker


def detect_faces(frame, min_area: float = MIN_FACE_AREA) -> list[Face]:
    """Faces in a BGR frame, biggest first."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(shrink(frame, DETECT_MAX_SIDE), cv2.COLOR_BGR2RGB)
    result = get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    faces = []
    for landmarks, shapes in zip(result.face_landmarks, result.face_blendshapes):
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if width * height >= min_area:
            faces.append(Face(
                cx=float((min(xs) + max(xs)) / 2), width=float(width), area=float(width * height),
                scores={c.category_name: float(c.score) for c in shapes},
            ))
    return sorted(faces, key=lambda f: f.area, reverse=True)


# ---------- expression rules ----------
def score_emotions(bs: dict) -> dict:
    """bs: blendshape scores. Returns {emotion: 0-1}. Tune the weights as needed."""
    g = lambda n: bs.get(n, 0.0)
    avg = lambda a, b: (g(a) + g(b)) / 2
    smile_l, smile_r = g("mouthSmileLeft"), g("mouthSmileRight")
    return {
        "happy": avg("mouthSmileLeft", "mouthSmileRight"),
        "smirk": max(smile_l, smile_r) if abs(smile_l - smile_r) > 0.25 else 0.0,
        "surprise": 0.4 * avg("eyeWideLeft", "eyeWideRight") + 0.3 * g("jawOpen") + 0.3 * g("browInnerUp"),
        "anger": 0.6 * avg("browDownLeft", "browDownRight")
                 + 0.2 * avg("noseSneerLeft", "noseSneerRight")
                 + 0.2 * avg("mouthPressLeft", "mouthPressRight"),
        "sad": 0.6 * avg("mouthFrownLeft", "mouthFrownRight") + 0.4 * g("browInnerUp"),
    }


def classify(bs: dict) -> tuple[str, float]:
    label, value = max(score_emotions(bs).items(), key=lambda kv: kv[1])
    return ("neutral" if value < EMOTION_THRESHOLD else label), round(value, 2)


# ---------- tracking across the sampled frames of one segment ----------
def link_tracks(samples: list[list[Face]]) -> list[list[Face | None]]:
    """Match faces across samples by horizontal position. Each track has one entry per sample."""
    n = len(samples)
    tracks: list[list[Face | None]] = []
    for k, faces in enumerate(samples):
        used = set()
        for face in faces:
            best, best_dist = None, TRACK_MATCH
            for ti, track in enumerate(tracks):
                if ti in used:
                    continue
                last = next((f for f in reversed(track[:k]) if f is not None), None)
                if last is not None and abs(face.cx - last.cx) < best_dist:
                    best, best_dist = ti, abs(face.cx - last.cx)
            if best is None:
                tracks.append([None] * n)
                best = len(tracks) - 1
            tracks[best][k] = face
            used.add(best)
    return tracks


def _expression(track):
    """(emotion, intensity, shift) for one track, or None."""
    labels = [classify(f.scores) if f is not None else None for f in track]
    found = [lab for lab in labels if lab]
    if not found:
        return None
    pool = [lab for lab in found if lab[0] != "neutral"] or found
    emotion = Counter(e for e, _ in pool).most_common(1)[0][0]
    intensity = max(v for e, v in pool if e == emotion)
    shift = bool(labels[0] and labels[-1] and labels[0][0] != labels[-1][0])
    return emotion, intensity, shift


def _mean_area(track) -> float:
    seen = [f.area for f in track if f is not None]
    return sum(seen) / len(seen) if seen else 0.0


def _jaw_movement(track) -> float:
    values = [f.scores.get("jawOpen", 0.0) for f in track if f is not None]
    return max(values) - min(values) if len(values) >= 2 else 0.0


def summarise(samples: list[list[Face]]):
    """Returns (face, listener). face: the main person {count, emotion, intensity, shift}, the one whose mouth
    moves when several are visible, otherwise the biggest. listener: the strongest non-neutral expression among
    the OTHER people in frame {emotion, intensity}, or None."""
    tracks = [t for t in link_tracks(samples) if any(f is not None for f in t)]
    if not tracks:
        return None, None

    main = max(tracks, key=_mean_area)
    if len(tracks) > 1:
        talker = max(tracks, key=_jaw_movement)
        if _jaw_movement(talker) >= MIN_JAW_MOVEMENT:
            main = talker

    face = None
    expression = _expression(main)
    if expression:
        emotion, intensity, shift = expression
        face = {"count": max(len(s) for s in samples), "emotion": emotion,
                "intensity": round(intensity, 2), "shift": shift}

    best_emotion, best_value = None, 0.0
    for track in tracks:
        if track is main:
            continue
        other = _expression(track)
        if other and other[0] != "neutral" and other[1] > best_value:
            best_emotion, best_value = other[0], other[1]
    listener = {"emotion": best_emotion, "intensity": round(best_value, 2)} if best_emotion else None
    return face, listener


# ---------- per-segment pass ----------
def _sample_times(seg) -> list[float]:
    dur = seg["end"] - seg["start"]
    mid = (seg["start"] + seg["end"]) / 2
    return [mid] if dur < 0.6 else [seg["start"] + 0.15 * dur, mid, seg["end"] - 0.15 * dur]


def add_face_signals(segments, video_path):
    """Adds face {count, emotion, intensity, shift} and listener {emotion, intensity}; each None when absent."""
    bar = Bar("faces", len(segments) or 1)
    with open_video(video_path) as cap:
        for i, seg in enumerate(segments):
            samples = []
            for t in _sample_times(seg):
                frame = read_frame_at(cap, t)
                samples.append(detect_faces(frame) if frame is not None else [])
            seg["face"], seg["listener"] = summarise(samples)
            bar.update(i + 1)
    bar.finish()

    seen = sum(1 for s in segments if s.get("face"))
    ok(f"Faces found in {seen}/{len(segments)} segments")
    return segments
