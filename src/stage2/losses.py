"""Stage 2 loss: L_tot + w_t L_tired + w_m L_mov + lambda L_res (all session level)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def ccc_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x - mx) * (y - my)).mean()
    return 2 * cov / (vx + vy + (mx - my) ** 2 + 1e-8)


def lds_weights(y_train: np.ndarray, sigma: float, max_score: int = 24) -> dict[int, float]:
    """Label-distribution smoothing: weight ∝ 1 / smoothed density, normalized to mean 1."""
    hist = np.bincount(y_train.astype(int), minlength=max_score + 1).astype(float)
    dens = gaussian_filter1d(hist, sigma=sigma, mode="constant") + 1e-6
    w = 1.0 / dens
    w = w / w[y_train.astype(int)].mean()
    return {i: float(w[i]) for i in range(max_score + 1)}


def stage2_loss(out: dict, y: torch.Tensor, tired3: torch.Tensor, mov: torch.Tensor, omega: torch.Tensor,
                pos_weight: torch.Tensor, lc) -> tuple[torch.Tensor, dict]:
    hub = F.huber_loss(out["y_tot"] / 24, y / 24, delta=lc.huber_delta, reduction="none")
    l_tot = (omega * hub).sum() / omega.sum()
    if len(y) > 1:
        l_tot = l_tot + lc.ccc_weight * (1 - ccc_torch(out["y_tot"], y))

    tgt = torch.stack([(tired3 > 0).float(), (tired3 > 1).float()], -1)
    l_tired = F.binary_cross_entropy_with_logits(out["tired_logits"], tgt)
    l_mov = F.binary_cross_entropy_with_logits(out["mov_logit"], mov, pos_weight=pos_weight)
    l_res = (out["deltas"] ** 2).sum(-1).mean()

    loss = l_tot + lc.w_tired * l_tired + lc.w_mov * l_mov + lc.lambda_res * l_res
    return loss, {"l_tot": l_tot.item(), "l_tired": l_tired.item(), "l_mov": l_mov.item(), "l_res": l_res.item()}
