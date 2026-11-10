"""Metrics (Section 8)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score


def ccc(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) < 2:
        return float("nan")
    my, mp = y.mean(), yhat.mean()
    cov = ((y - my) * (yhat - mp)).mean()
    return float(2 * cov / (y.var() + yhat.var() + (my - mp) ** 2 + 1e-12))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def binned_mae(y, yhat, edges) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y < hi)
        out[f"{lo}-{hi - 1}"] = mae(y[m], yhat[m]) if m.any() else float("nan")
    return out


def qwk(y, yhat_int) -> float:
    return float(cohen_kappa_score(np.asarray(y, int), np.asarray(yhat_int, int), weights="quadratic"))


def safe_auc(y, p) -> tuple[float, float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def total_report(y, yhat, edges) -> dict:
    return {"ccc": ccc(y, yhat), "rmse": rmse(y, yhat), "mae": mae(y, yhat), "n": int(len(y)),
            "binned_mae": binned_mae(y, yhat, edges)}


def item_report(y_items: np.ndarray, probs: np.ndarray) -> dict:
    """y_items [N, 8]; probs [N, 8, 4] -> per-item MAE (expected score) and QWK (argmax)."""
    exp = (probs * np.arange(probs.shape[-1])).sum(-1)
    arg = probs.argmax(-1)
    return {f"item{k + 1}": {"mae": mae(y_items[:, k], exp[:, k]), "qwk": qwk(y_items[:, k], arg[:, k])}
            for k in range(y_items.shape[1])}
