"""Session-level dataset: each item = one session's 8 prompts (so L_sum can be computed)."""
from __future__ import annotations

import random

import numpy as np

from .. import N_ITEMS
from ..config import Cfg
from ..preprocessing.pipeline import load_qa_pairs
from .prompts import build_bin_prompt, build_prompt, pair_lines
from .retrieval import load_item_defs


def bin_enabled(cfg: Cfg) -> bool:
    return bool(cfg.stage1.get("severity", {}).get("enabled", False))


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
        prompts, used = [], set()
        for k in range(1, N_ITEMS + 1):
            r = self.ret[sid][k]
            chosen = self._select(r["ranked"])
            used.update(chosen)
            pairs = [self.pairs[sid][pid] for pid in chosen]
            prompts.append(build_prompt(k, self.items[k]["text"], pairs, r["ev"],
                                        self.cfg.stage1.model.max_answer_words))
        out = {"sid": sid, "prompts": prompts}
        if bin_enabled(self.cfg):           # 区间题的证据 = 8 道题用到的 Q-A 对的并集（按时间）
            out["bin_prompt"] = build_bin_prompt(pair_lines([self.pairs[sid][p] for p in sorted(used)],
                                                            self.cfg.stage1.model.max_answer_words))
        if self.labels is not None:
            row = self.labels.loc[sid]
            out["y_items"] = np.array([row[f"item{k}"] for k in range(1, N_ITEMS + 1)], dtype=np.int64)
            out["y_total"] = float(row["phq_total"])
        return out
