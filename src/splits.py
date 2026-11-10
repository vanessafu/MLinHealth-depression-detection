"""Stratified K-fold split over training sessions (shared by Stage 1 and Stage 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import Cfg, resolve


def _strata(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def make_folds(lab: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    sc = cfg.splits
    tr = lab[lab["split"] == "train"].copy()
    tr["phq_bin"] = np.digitize(tr["phq_total"], sc.phq_bin_edges[1:-1])
    cols = list(sc.stratify_by)
    # 某层人数少于折数时，从后往前去掉分层变量
    while cols:
        strata = _strata(tr, cols)
        if strata.value_counts().min() >= sc.n_folds:
            break
        cols = cols[:-1]
    strata = _strata(tr, cols) if cols else pd.Series("all", index=tr.index)
    skf = StratifiedKFold(n_splits=sc.n_folds, shuffle=True, random_state=sc.seed)
    tr["fold"] = -1
    for f, (_, idx) in enumerate(skf.split(tr, strata)):
        tr.iloc[idx, tr.columns.get_loc("fold")] = f
    tr.attrs["stratify_cols_used"] = cols
    return tr[["sid", "fold", "phq_total", "gender", "mode"]].reset_index(drop=True)


def folds_path(cfg: Cfg) -> Path:
    return resolve(cfg.paths.splits) / "folds.csv"


def load_folds(cfg: Cfg) -> pd.DataFrame:
    p = folds_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，先运行 scripts/preprocessing/prepare_labels.py")
    return pd.read_csv(p)


def fold_ids(folds: pd.DataFrame, f: int) -> tuple[list[int], list[int]]:
    """(train sids, val sids) for fold f."""
    return folds.loc[folds.fold != f, "sid"].tolist(), folds.loc[folds.fold == f, "sid"].tolist()
