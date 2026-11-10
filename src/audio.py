"""Audio IO helpers."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import Cfg, resolve


def find_audio(cfg: Cfg) -> dict[int, Path]:
    """Map session id -> wav path, searching paths.raw_audio with paths.audio_glob."""
    root = resolve(cfg.paths.raw_audio)
    out = {}
    for p in sorted(root.glob(cfg.paths.audio_glob)):
        m = re.match(r"(\d+)_AUDIO", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


def load_wav(path: Path, sr: int = 16000) -> np.ndarray:
    """Mono float32 waveform at `sr`."""
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, file_sr)
        wav = resample_poly(wav, sr // g, file_sr // g).astype(np.float32)
    return wav
