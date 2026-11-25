"""PHQ-8 严重程度区间 + 由 8 题分布推出的总分分布。

区间（默认 4 个，>=15 合并，因为 20-24 在 train 里只有 7 人）：0-4 / 5-9 / 10-14 / 15-24
  sum_distribution: 8 个独立的 [4] 分布卷积 -> 总分 0..24 的分布 [25]
  bin_from_sum:     总分分布 -> 区间分布
  combine:          专家乘积 P(s) ∝ P_items(s) · P_head(bin(s))^gamma，返回期望总分
"""
from __future__ import annotations

import numpy as np
import torch

MAX_TOTAL = 24
DEFAULT_EDGES = [0, 5, 10, 15, 25]


def bin_index(totals, edges=DEFAULT_EDGES):
    """总分 -> 区间号 0..len(edges)-2。"""
    return np.clip(np.digitize(np.asarray(totals, dtype=float), edges[1:-1]), 0, len(edges) - 2)


def _bin_matrix(edges, device=None, dtype=torch.float32):
    s = np.arange(MAX_TOTAL + 1)
    b = bin_index(s, edges)
    m = np.zeros((MAX_TOTAL + 1, len(edges) - 1))
    m[s, b] = 1
    return torch.tensor(m, device=device, dtype=dtype)


def sum_distribution(p: torch.Tensor) -> torch.Tensor:
    """p [S, 8, 4] -> [S, 25]（可微）。"""
    d = p[:, 0]
    for k in range(1, p.shape[1]):
        L = d.shape[1]
        out = d.new_zeros(d.shape[0], L + p.shape[2] - 1)
        for c in range(p.shape[2]):
            out[:, c:c + L] = out[:, c:c + L] + d * p[:, k, c:c + 1]
        d = out
    return d


def bin_from_sum(dist: torch.Tensor, edges=DEFAULT_EDGES) -> torch.Tensor:
    return dist @ _bin_matrix(edges, dist.device, dist.dtype)


def sum_distribution_np(p: np.ndarray) -> np.ndarray:
    return sum_distribution(torch.tensor(np.asarray(p), dtype=torch.float64)).numpy()


def combine(item_probs: np.ndarray, bin_probs: np.ndarray | None, edges=DEFAULT_EDGES,
            gamma: float = 1.0) -> np.ndarray:
    """期望总分；bin_probs 为 None 时就是 8 题期望之和。"""
    if bin_probs is None:
        return (np.asarray(item_probs) * np.arange(item_probs.shape[-1])).sum(-1).sum(-1)
    d = sum_distribution_np(item_probs)                                   # [N, 25]
    b = bin_index(np.arange(MAX_TOTAL + 1), edges)
    post = d * np.clip(np.asarray(bin_probs, dtype=float)[:, b], 1e-6, None) ** gamma
    post /= post.sum(1, keepdims=True)
    return (post * np.arange(MAX_TOTAL + 1)).sum(1)
