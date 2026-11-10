"""Preprocessing quality report (Section 3 checklist)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import N_ITEMS
from ..config import Cfg
from . import interim_path


def qc_report(cfg: Cfg, sids: list[int]) -> pd.DataFrame:
    rows = []
    for sid in sids:
        p_qa, p_seg = interim_path(cfg, "qa_pairs", sid, "json"), interim_path(cfg, "segments", sid, "csv")
        if not p_qa.exists():
            rows.append({"sid": sid, "status": "missing_qa"})
            continue
        qa = json.load(open(p_qa))
        pairs = qa["pairs"]
        dur = np.array([p["t_end"] - p["t_start"] for p in pairs]) if pairs else np.zeros(1)
        n_seg = len(pd.read_csv(p_seg)) if p_seg.exists() else np.nan
        mapped = {k for p in pairs for k in p["items"]}
        row = {"sid": sid, "status": "ok", "ellie_speaker": qa["ellie_speaker"],
               "n_pairs": len(pairs), "n_segments": n_seg,
               "answer_sec_median": float(np.median(dur)), "answer_sec_max": float(dur.max()),
               "n_answers_lt1s": int((dur < cfg.preprocessing.segments.min_sec).sum())}
        for k in range(1, N_ITEMS + 1):
            row[f"map_hit_item{k}"] = int(k in mapped)
        rows.append(row)
    return pd.DataFrame(rows)
