"""Assemble final predictions (beta fusion) and compute all Section 8 metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import N_ITEMS
from ..config import Cfg
from ..fusion import fuse, select_beta
from ..labels import ITEM_COLS
from ..stage1.teacher import teacher_features
from .metrics import item_report, mae, qwk, rmse, safe_auc, total_report


def build_final(pred: pd.DataFrame, labels: pd.DataFrame, beta_grid: list[float]) -> tuple[pd.DataFrame, dict]:
    """pred columns: sid, split, y_text, y_tot (+ optional tired_expected, mov_prob)."""
    df = pred.merge(labels[["sid", "phq_total"]], on="sid", how="left")
    tr = df[df.split == "train"]
    beta, scores = select_beta(tr.phq_total.to_numpy(), tr.y_text.to_numpy(), tr.y_tot.to_numpy(), beta_grid)
    df["y_final"] = fuse(df.y_text.to_numpy(), df.y_tot.to_numpy(), beta)
    return df.drop(columns="phq_total"), {"beta": beta, "beta_scores": scores}


def evaluate(final: pd.DataFrame, labels: pd.DataFrame, teacher: dict, cfg: Cfg) -> dict:
    edges = cfg.evaluation.severity_bin_edges
    lab = labels.set_index("sid")
    train_mean = float(lab.loc[lab.split == "train", "phq_total"].mean())
    out = {}
    for split, g in final.groupby("split"):
        g = g[g.sid.isin(lab.index[lab.phq_total.notna()])]
        if g.empty:
            continue
        y = lab.loc[g.sid, "phq_total"].to_numpy()
        rep = {c: total_report(y, g[c].to_numpy(), edges) for c in ["y_text", "y_tot", "y_final"] if c in g}
        rep["mean_baseline_rmse"] = rmse(y, np.full_like(y, train_mean, dtype=float))
        if cfg.evaluation.report_ai_subset and split == "dev":
            ai = (lab.loc[g.sid, "mode"] == "ai").to_numpy()
            if ai.any():
                rep["ai_subset"] = total_report(y[ai], g.y_final.to_numpy()[ai], edges)

        # Stage 1 逐题
        tm = teacher["split"] == split
        tsids = teacher["sids"][tm]
        has = [s for s in tsids if s in lab.index and lab.loc[s, ITEM_COLS].notna().all()]
        if has:
            pos = {s: i for i, s in enumerate(tsids)}
            probs = teacher["probs"][tm][[pos[s] for s in has]]
            rep["items"] = item_report(lab.loc[has, ITEM_COLS].to_numpy().astype(int), probs)

        # 辅助任务
        if "tired_expected" in g and lab.loc[g.sid, "tired3"].notna().all():
            t = lab.loc[g.sid, "tired3"].to_numpy()
            te = g.tired_expected.to_numpy()
            rep["tired3"] = {"qwk": qwk(t, np.clip(np.rint(te), 0, 2)), "mae": mae(t, te)}
        if "mov_prob" in g and lab.loc[g.sid, "mov"].notna().all():
            auroc, auprc = safe_auc(lab.loc[g.sid, "mov"].to_numpy(), g.mov_prob.to_numpy())
            rep["moving"] = {"auroc": auroc, "auprc": auprc}
        out[split] = rep
    return out


def text_only_predictions(teacher: dict) -> pd.DataFrame:
    tf = teacher_features(teacher["probs"])
    return pd.DataFrame({"sid": teacher["sids"], "split": teacher["split"], "y_text": tf["y_text"],
                         "y_tot": tf["y_text"]})


assert N_ITEMS == 8
