"""Residual MIL model (Section 6)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMIL(nn.Module):
    def __init__(self, n_layers: int = 25, feat_dim: int = 2048, proj_dim: int = 256, attn_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.3, max_delta: float = 6.0,
                 pooling: str = "attention", use_pos: bool = True, n_teacher: int = 4):
        super().__init__()
        self.max_delta, self.pooling, self.use_pos = max_delta, pooling, use_pos
        self.alpha = nn.Parameter(torch.zeros(n_layers))                       # 层权重
        self.proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, proj_dim), nn.GELU(),
                                  nn.Dropout(dropout))
        a_in = proj_dim + int(use_pos)
        self.attn_V = nn.Linear(a_in, attn_dim)
        self.attn_U = nn.Linear(a_in, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.trunk = nn.Sequential(nn.Linear(proj_dim + n_teacher, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.Linear(hidden_dim, 3)                                  # Δ_tot, Δ_tired, Δ_mov
        nn.init.zeros_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)

    def forward(self, feats: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor,
                y_text: torch.Tensor, tau: torch.Tensor, nu: torch.Tensor) -> dict:
        """feats [B, M, L, D]; pos/mask [B, M]; y_text [B]; tau [B, 2]; nu [B]."""
        lw = self.alpha.softmax(0)
        x = (feats.float() * lw.view(1, 1, -1, 1)).sum(2)                      # [B, M, D]
        e = self.proj(x)                                                       # [B, M, P]

        if self.pooling == "attention":
            a_in = torch.cat([e, pos.unsqueeze(-1)], -1) if self.use_pos else e
            s = self.attn_w(torch.tanh(self.attn_V(a_in)) * torch.sigmoid(self.attn_U(a_in))).squeeze(-1)
        else:                                                                  # 均值池化（消融 A9）
            s = torch.zeros(mask.shape, device=e.device)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.nan_to_num(s.softmax(-1), nan=0.0)                           # 空包时全 0
        H = (a.unsqueeze(-1) * e).sum(1)                                       # [B, P]

        c = torch.stack([y_text / 24, torch.sigmoid(tau[:, 0]), torch.sigmoid(tau[:, 1]),
                         torch.sigmoid(nu)], -1)
        d = self.heads(self.trunk(torch.cat([H, c], -1)))
        d_tot, d_tired, d_mov = d[:, 0], d[:, 1], d[:, 2]

        y_tot = (y_text + self.max_delta * torch.tanh(d_tot)).clamp(0, 24)
        return {"y_tot": y_tot, "tired_logits": tau + d_tired.unsqueeze(-1), "mov_logit": nu + d_mov,
                "deltas": d, "attn": a, "layer_weights": lw}
