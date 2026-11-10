"""Final fusion: y_final = y_text + beta * (y_tot - y_text); beta chosen on train OOF by CCC."""
from __future__ import annotations

import numpy as np

from .evaluation.metrics import ccc


def fuse(y_text: np.ndarray, y_tot: np.ndarray, beta: float) -> np.ndarray:
    return np.clip(y_text + beta * (y_tot - y_text), 0, 24)


def select_beta(y: np.ndarray, y_text: np.ndarray, y_tot: np.ndarray, grid: list[float]) -> tuple[float, dict]:
    scores = {float(b): ccc(y, fuse(y_text, y_tot, b)) for b in grid}
    best = max(scores, key=scores.get)
    return best, scores
