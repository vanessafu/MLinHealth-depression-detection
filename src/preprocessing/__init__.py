"""Offline preprocessing: wav -> Q-A pairs + participant segments + WavLM feature cache."""
from __future__ import annotations

from pathlib import Path

from ..config import Cfg, resolve


def qa_variant(cfg: Cfg) -> str:
    """Q-A pairs depend on keep_rapport, so they live in separate sub-folders."""
    return "rapport" if cfg.preprocessing.qa.keep_rapport else "norapport"


def interim_path(cfg: Cfg, kind: str, sid: int, ext: str) -> Path:
    base = resolve(cfg.paths.interim) / kind
    if kind in ("qa_pairs", "segments"):
        base = base / qa_variant(cfg)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sid}.{ext}"


def feature_path(cfg: Cfg, sid: int) -> Path:
    base = resolve(cfg.paths.features) / "wavlm" / qa_variant(cfg)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sid}.pt"
