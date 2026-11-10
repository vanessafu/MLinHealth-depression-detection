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


def stage1_loss(logits: torch.Tensor, y_items: torch.Tensor, y_total: torch.Tensor, w: torch.Tensor,
                mu: float, lambda_sum: float) -> tuple[torch.Tensor, dict]:
    """logits [S, 8, 4]; y_items [S, 8] long; y_total [S]; w [8, 4]."""
    logp = F.log_softmax(logits, dim=-1)
    yhat = expected_score(logits)                                      # [S, 8]
    ce = -logp.gather(-1, y_items.unsqueeze(-1)).squeeze(-1)           # [S, 8]
    se = (yhat - y_items.float()) ** 2
    wk = w.to(logits.device)[torch.arange(N_ITEMS, device=logits.device), y_items]  # [S, 8]
    l_item = (wk * (ce + mu * se)).mean(-1)                            # [S]
    l_sum = (yhat.sum(-1) - y_total) ** 2 / N_ITEMS                    # [S]
    loss = (l_item + lambda_sum * l_sum).mean()
    return loss, {"l_item": l_item.mean().item(), "l_sum": l_sum.mean().item()}
