"""Step 2: Silero VAD. Speech intervals on the original timeline (seconds)."""
from __future__ import annotations

import numpy as np

from ..config import Cfg

_MODEL = None


def run_vad(wav: np.ndarray, cfg: Cfg) -> list[tuple[float, float]]:
    global _MODEL
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _MODEL is None:
        _MODEL = load_silero_vad()
    vc, sr = cfg.preprocessing.vad, cfg.preprocessing.sample_rate
    ts = get_speech_timestamps(
        torch.from_numpy(wav), _MODEL, sampling_rate=sr, threshold=vc.threshold,
        min_speech_duration_ms=vc.min_speech_ms, min_silence_duration_ms=vc.min_silence_ms,
        speech_pad_ms=vc.speech_pad_ms, return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def mask_non_speech(wav: np.ndarray, intervals: list[tuple[float, float]], sr: int) -> np.ndarray:
    """Zero out non-speech so Whisper cannot hallucinate there, keeping the original timeline."""
    out = np.zeros_like(wav)
    for s, e in intervals:
        a, b = int(s * sr), int(e * sr)
        out[a:b] = wav[a:b]
    return out
