"""Session bags of cached WavLM features + teacher inputs, kept in memory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import Cfg
from ..preprocessing import feature_path
from ..stage1.teacher import load_teacher, teacher_features


class Bags:
    def __init__(self, cfg: Cfg, sids: list[int], teacher_path, labels: pd.DataFrame | None = None):
        t = load_teacher(teacher_path)
        idx = {int(s): i for i, s in enumerate(t["sids"])}
        missing = [s for s in sids if s not in idx]
        if missing:
            raise KeyError(f"teacher predictions missing for sessions: {missing[:10]}")
        tf = teacher_features(t["probs"][[idx[s] for s in sids]])
        self.sids = list(sids)
        self.y_text = torch.tensor(tf["y_text"], dtype=torch.float32)
        self.tau = torch.tensor(tf["tau"], dtype=torch.float32)
        self.nu = torch.tensor(tf["nu"], dtype=torch.float32)
        self.feats, self.pos = [], []
        for s in self.sids:
            d = torch.load(feature_path(cfg, s))
            self.feats.append(d["feats"])        # [M, L, D] fp16
            self.pos.append(d["pos"])
        self.n_layers, self.feat_dim = self.feats[0].shape[1:]
        if labels is not None:
            lab = labels.set_index("sid").loc[self.sids]
            self.y = torch.tensor(lab["phq_total"].to_numpy(), dtype=torch.float32)
            self.tired3 = torch.tensor(lab["tired3"].to_numpy(), dtype=torch.float32)
            self.mov = torch.tensor(lab["mov"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sids)

    def batch(self, idx: list[int], max_segments: int | None, rng: np.random.Generator | None,
              device: str) -> dict:
        """Pad bags to the longest in the batch; optionally subsample <= max_segments per bag."""
        chosen = []
        for i in idx:
            m = len(self.feats[i])
            sel = np.arange(m)
            if max_segments is not None and m > max_segments and rng is not None:
                sel = np.sort(rng.choice(m, size=max_segments, replace=False))
            chosen.append(sel)
        M = max(1, max(len(c) for c in chosen))
        B = len(idx)
        feats = torch.zeros(B, M, self.n_layers, self.feat_dim, dtype=torch.float16)
        pos = torch.zeros(B, M)
        mask = torch.zeros(B, M, dtype=torch.bool)
        for b, (i, sel) in enumerate(zip(idx, chosen)):
            n = len(sel)
            if n:
                feats[b, :n] = self.feats[i][sel]
                pos[b, :n] = self.pos[i][sel]
                mask[b, :n] = True
        out = {"feats": feats, "pos": pos, "mask": mask, "y_text": self.y_text[idx], "tau": self.tau[idx],
               "nu": self.nu[idx]}
        if hasattr(self, "y"):
            out.update(y=self.y[idx], tired3=self.tired3[idx], mov=self.mov[idx])
        return {k: v.to(device) for k, v in out.items()}
