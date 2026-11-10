"""Teacher predictions p~_ik saved by Stage 1 and consumed by Stage 2.

File: <stage1_dir>/teacher_<kind>.npz with arrays
  sids  [N] int, split [N] str, probs [N, 8, 4] float32
kind: oof (train = OOF, dev/test = mean over fold models) | in_sample (ablation A8) | zeroshot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import MOVING_IDX, N_CLASSES, TIRED_IDX

EPS = 1e-4


def save_teacher(path: Path, sids, split, probs) -> None:
    np.savez(path, sids=np.asarray(sids, dtype=np.int64), split=np.asarray(split, dtype=str),
             probs=np.asarray(probs, dtype=np.float32))


def load_teacher(path: Path) -> dict:
    d = np.load(path)
    return {"sids": d["sids"], "split": d["split"], "probs": d["probs"]}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def teacher_features(probs: np.ndarray) -> dict:
    """probs [N, 8, 4] -> y_text [N], tau [N, 2], nu [N], item_scores [N, 8]."""
    item_scores = (probs * np.arange(N_CLASSES)).sum(-1)
    pt, pm = probs[:, TIRED_IDX], probs[:, MOVING_IDX]
    tau = np.stack([_logit(1 - pt[:, 0]), _logit(pt[:, 2] + pt[:, 3])], axis=1)  # P(>0), P(>1) on 3 levels
    nu = _logit(1 - pm[:, 0])
    return {"y_text": item_scores.sum(-1), "tau": tau, "nu": nu, "item_scores": item_scores}
