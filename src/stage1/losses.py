"""Stage 1 loss: L_item = w_k(y)[CE + mu*(E[y]-y)^2], L_sum = (sum_k E[y_k] - y)^2 / 8."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .. import N_CLASSES, N_ITEMS


def class_weights(y_items: np.ndarray, mode: str = "inv_sqrt_freq", smoothing: float = 1.0) -> torch.Tensor:
    """[8, 4] weights from the training fold; normalized so the mean weight over training samples is 1."""
    if mode == "none":
        return torch.ones(N_ITEMS, N_CLASSES)
    w = np.zeros((N_ITEMS, N_CLASSES))
    for k in range(N_ITEMS):
        cnt = np.bincount(y_items[:, k].astype(int), minlength=N_CLASSES) + smoothing
        wk = 1.0 / np.sqrt(cnt / cnt.sum())
        wk /= wk[y_items[:, k].astype(int)].mean()
        w[k] = wk
    return torch.tensor(w, dtype=torch.float32)


def expected_score(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(-1)
    return (p * torch.arange(N_CLASSES, device=logits.device, dtype=p.dtype)).sum(-1)


def session_weights(y_total_train: np.ndarray, mode: str = "none", sigma: float = 2.0,
                    power: float = 0.5) -> dict[int, float]:
    """按总分给 session 加权（高分样本少 -> 权重大）：w(y) ∝ (平滑后的密度)^-power，train 上均值为 1。"""
    from scipy.ndimage import gaussian_filter1d

    if mode in (None, "none"):
        return {i: 1.0 for i in range(25)}
    hist = np.bincount(np.asarray(y_total_train).astype(int), minlength=25).astype(float)
    dens = gaussian_filter1d(hist, sigma=sigma, mode="constant") + 1e-3
    w = dens ** (-power)
    w = w / w[np.asarray(y_total_train).astype(int)].mean()
    return {i: float(w[i]) for i in range(25)}


def stage1_loss(logits: torch.Tensor, y_items: torch.Tensor, y_total: torch.Tensor, w: torch.Tensor,
                mu: float, lambda_sum: float, sess_w: torch.Tensor | None = None, lambda_bin: float = 0.0,
                bin_edges=None) -> tuple[torch.Tensor, dict]:
    """logits [S, 8, 4]; y_items [S, 8] long; y_total [S]; w [8, 4]; sess_w [S] 样本权重。
    lambda_bin > 0：加区间一致性项 -log P_items(总分落在真实区间)，P 由 8 题分布卷积得到。"""
    logp = F.log_softmax(logits, dim=-1)
    yhat = expected_score(logits)                                      # [S, 8]
    ce = -logp.gather(-1, y_items.unsqueeze(-1)).squeeze(-1)           # [S, 8]
    se = (yhat - y_items.float()) ** 2
    wk = w.to(logits.device)[torch.arange(N_ITEMS, device=logits.device), y_items]  # [S, 8]
    l_item = (wk * (ce + mu * se)).mean(-1)                            # [S]
    l_sum = (yhat.sum(-1) - y_total) ** 2 / N_ITEMS                    # [S]
    per = l_item + lambda_sum * l_sum
    parts = {"l_item": l_item.mean().item(), "l_sum": l_sum.mean().item()}
    if lambda_bin > 0:
        from .severity import DEFAULT_EDGES, bin_from_sum, bin_index, sum_distribution

        edges = bin_edges or DEFAULT_EDGES
        pb = bin_from_sum(sum_distribution(logp.exp()), edges)          # [S, n_bins]
        tb = torch.tensor(bin_index(y_total.detach().cpu().numpy(), edges), device=logits.device)
        l_bin = -torch.log(pb.gather(1, tb[:, None]).squeeze(1).clamp_min(1e-6))
        per = per + lambda_bin * l_bin
        parts["l_bin_cons"] = l_bin.mean().item()
    if sess_w is not None:   # 权重在整个 train 上均值为 1，不能在 batch 内归一化（micro-batch = 1 时会被约掉）
        loss = (sess_w * per).mean()
    else:
        loss = per.mean()
    return loss, parts
