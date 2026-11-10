"""Session-level dataset: each item = one session's 8 prompts (so L_sum can be computed)."""
from __future__ import annotations

import random

import numpy as np

from .. import N_ITEMS
from ..config import Cfg
from ..preprocessing.pipeline import load_qa_pairs
from .prompts import build_prompt
from .retrieval import load_item_defs


class Stage1Sessions:
    def __init__(self, cfg: Cfg, sids: list[int], retrieval: dict, labels=None, train: bool = False,
                 seed: int = 0):
        self.cfg, self.sids, self.ret, self.train = cfg, list(sids), retrieval, train
        self.items = load_item_defs(cfg)
        self.pairs = {sid: {p["pair_id"]: p for p in load_qa_pairs(cfg, sid)} for sid in self.sids}
        self.labels = labels.set_index("sid") if labels is not None else None
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sids)

    def _select(self, ranked: list[int]) -> list[int]:
        ac, rc = self.cfg.stage1.augment, self.cfg.stage1.retrieval
        if not self.train:
            chosen = ranked[: rc.top_n]
            return sorted(chosen)                                  # 按访谈时间排序
        n = self.rng.randint(ac.n_min, ac.n_max)
        chosen = ranked[:n]
        if len(chosen) > 1 and self.rng.random() < ac.drop_one_prob:
            drop = self.rng.randrange(len(chosen))
            chosen = [c for i, c in enumerate(chosen) if i != drop]
        chosen = sorted(chosen)
        if self.rng.random() < ac.shuffle_prob:
            self.rng.shuffle(chosen)
        return chosen

    def get(self, idx: int) -> dict:
        sid = self.sids[idx]
        prompts = []
        for k in range(1, N_ITEMS + 1):
            r = self.ret[sid][k]
            pairs = [self.pairs[sid][pid] for pid in self._select(r["ranked"])]
            prompts.append(build_prompt(k, self.items[k]["text"], pairs, r["ev"],
                                        self.cfg.stage1.model.max_answer_words))
        out = {"sid": sid, "prompts": prompts}
        if self.labels is not None:
            row = self.labels.loc[sid]
            out["y_items"] = np.array([row[f"item{k}"] for k in range(1, N_ITEMS + 1)], dtype=np.int64)
            out["y_total"] = float(row["phq_total"])
        return out
