"""Load E-DAIC labels into one tidy table (one row per session)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import MOVING_IDX, N_ITEMS, TIRED_IDX
from .config import Cfg, resolve

ITEM_COLS = [f"item{k}" for k in range(1, N_ITEMS + 1)]


def acquisition_mode(sid: int, ai_range: list[int]) -> str:
    return "ai" if ai_range[0] <= sid <= ai_range[1] else "woz"


def load_labels(cfg: Cfg) -> pd.DataFrame:
    """Columns: sid, split, gender, mode, phq_total, phq_binary, item1..item8, tired3, mov."""
    lc = cfg.labels
    ldir = resolve(cfg.paths.labels_dir)
    frames = []
    for split, fname in lc.split_files.items():
        path = ldir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={lc.id_col: "sid", lc.total_col: "phq_total",
                                lc.binary_col: "phq_binary", lc.gender_col: "gender"})
        df["split"] = split
        keep = [c for c in ["sid", "split", "gender", "phq_total", "phq_binary"] if c in df.columns]
        frames.append(df[keep])
    if not frames:
        raise FileNotFoundError(f"no split files found in {ldir}")
    lab = pd.concat(frames, ignore_index=True)
    lab["sid"] = lab["sid"].astype(int)

    det_path = ldir / lc.detailed_file
    if det_path.exists():
        det = pd.read_csv(det_path).rename(columns={lc.id_col: "sid"})
        det = det.rename(columns=dict(zip(lc.item_cols, ITEM_COLS)))
        det["sid"] = det["sid"].astype(int)
        lab = lab.merge(det[["sid"] + ITEM_COLS], on="sid", how="left")
    else:
        for c in ITEM_COLS:
            lab[c] = np.nan

    lab["mode"] = [acquisition_mode(s, lc.ai_id_range) for s in lab["sid"]]
    lab["tired3"] = np.minimum(lab[f"item{TIRED_IDX + 1}"], 2)
    lab["mov"] = (lab[f"item{MOVING_IDX + 1}"] >= 1).astype(float)
    lab.loc[lab[f"item{MOVING_IDX + 1}"].isna(), "mov"] = np.nan
    return lab.sort_values("sid").reset_index(drop=True)


def item_matrix(lab: pd.DataFrame) -> np.ndarray:
    """[N, 8] integer item scores (NaN if unavailable)."""
    return lab[ITEM_COLS].to_numpy(dtype=float)


def consistency_report(lab: pd.DataFrame) -> dict:
    """Check sum(items) == PHQ_Score and PHQ_Binary == (score >= 10)."""
    rep = {"n_sessions": int(len(lab)), "per_split": lab["split"].value_counts().to_dict()}
    has_items = lab[ITEM_COLS].notna().all(axis=1) & lab["phq_total"].notna()
    s = lab[ITEM_COLS].sum(axis=1)                         # 全表求和，和 phq_total 同索引才能比较
    bad_sum = lab.loc[has_items & (s != lab["phq_total"]), "sid"].tolist()
    rep["sum_mismatch_sids"] = bad_sum
    if "phq_binary" in lab:
        ok = lab["phq_binary"].notna() & lab["phq_total"].notna()
        bad_bin = lab.loc[ok & ((lab["phq_total"] >= 10).astype(int) != lab["phq_binary"]), "sid"].tolist()
        rep["binary_mismatch_sids"] = bad_bin
    tr = lab[lab["split"] == "train"]
    rep["train_tired3_counts"] = tr["tired3"].value_counts().sort_index().to_dict()
    rep["train_mov_counts"] = tr["mov"].value_counts().sort_index().to_dict()
    # 键用 "split/mode" 字符串：元组键 json.dumps 会报错（prepare_labels.py 打印报告时崩溃）
    rep["mode_by_split"] = {f"{s}/{m}": int(n) for (s, m), n in lab.groupby(["split", "mode"]).size().items()}
    return rep
