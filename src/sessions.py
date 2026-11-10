"""Which sessions of a split have the required preprocessing outputs."""
from __future__ import annotations

import pandas as pd

from .config import Cfg
from .preprocessing import feature_path, interim_path


def available_sids(cfg: Cfg, labels: pd.DataFrame, split: str, need: tuple[str, ...] = ("qa",)) -> list[int]:
    sids = labels.loc[labels.split == split, "sid"].astype(int).tolist()
    ok = []
    for s in sids:
        if "qa" in need and not interim_path(cfg, "qa_pairs", s, "json").exists():
            continue
        if "wavlm" in need and not feature_path(cfg, s).exists():
            continue
        ok.append(s)
    if len(ok) < len(sids):
        print(f"[{split}] {len(sids) - len(ok)}/{len(sids)} sessions missing {need}, skipped")
    return ok
