"""Speech to text with word timings (faster-whisper).

Takes the audio already decoded by media.load_audio(), so the file is decoded once for the
whole job and faster-whisper never has to open the video itself.
"""
import os
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from app.pipeline.media import SAMPLE_RATE
from app.pipeline.progress import Bar, ok

# --- accuracy settings (all in one place) ---
BEAM_SIZE = 5
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)     # starts deterministic, falls back only when a window fails the checks
NO_SPEECH_THRESHOLD = 0.6                         # music / noise treated as non-speech
COMPRESSION_RATIO_THRESHOLD = 2.4                 # repetition loops are retried at a higher temperature
LOG_PROB_THRESHOLD = -1.0
HALLUCINATION_SILENCE = 2.0                       # skip silent gaps where Whisper invents words (needs word timestamps)
VAD = {"threshold": 0.4, "min_silence_duration_ms": 700, "speech_pad_ms": 300}   # lower threshold keeps speech under music

_models: dict[str, WhisperModel] = {}


@dataclass(frozen=True)
class Transcript:
    segments: list[dict]      # {start, end, text, words:[{w,s,e,p}], conf}
    language: str
    duration: float


def _device() -> tuple[str, str]:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _model(size: str) -> WhisperModel:
    """Loaded once per worker process and reused by every job."""
    if size not in _models:
        device, compute = _device()
        kwargs = {"cpu_threads": min(os.cpu_count() or 4, 8)} if device == "cpu" else {}
        with Bar(f"whisper: loading {size}", None):
            _models[size] = WhisperModel(size, device=device, compute_type=compute, **kwargs)
    return _models[size]


def _prompt(context, vocabulary) -> str | None:
    parts = []
    if context:
        parts.append(context.strip())
    if vocabulary:
        parts.append("Names and terms: " + ", ".join(vocabulary) + ".")
    return " ".join(parts) or None


def transcribe_audio(
    audio: np.ndarray,
    *,
    model_size: str = "small",
    language: str | None = "en",
    context: str | None = None,
    vocabulary: list[str] | None = None,
) -> Transcript:
    """audio: 16 kHz mono float32 from media.load_audio(). language None or "" = detect automatically."""
    if model_size.endswith(".en"):
        language = "en"
    vocabulary = vocabulary or []
    duration = len(audio) / SAMPLE_RATE

    # condition_on_previous_text=False stops one mistake from snowballing. The side effect is that
    # initial_prompt only guides the first window, so names are also passed as hotwords, which apply to every window.
    raw_segments, info = _model(model_size).transcribe(
        audio,
        language=language or None,
        beam_size=BEAM_SIZE,
        temperature=list(TEMPERATURES),
        condition_on_previous_text=False,
        initial_prompt=_prompt(context, vocabulary),
        hotwords=", ".join(vocabulary) or None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=VAD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        hallucination_silence_threshold=HALLUCINATION_SILENCE,
    )

    bar = Bar("whisper: transcribing", duration or 1.0)
    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if text and seg.end > seg.start:
            words = [
                {"w": w.word.strip(), "s": round(float(w.start), 2), "e": round(float(w.end), 2),
                 "p": round(float(w.probability), 2)}
                for w in (seg.words or [])
            ]
            probs = [w["p"] for w in words]
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "words": words,
                "conf": round(sum(probs) / len(probs), 2) if probs else None,
            })
        bar.update(min(float(seg.end), duration))
    bar.finish()

    ok(f"Transcribed {len(segments)} segments ({info.language})")
    return Transcript(segments=segments, language=info.language, duration=duration)
