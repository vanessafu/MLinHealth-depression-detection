"""比较试跑三种转写来源的 Stage 1 OOF 结果（outputs/trial_<variant>/stage1/teacher_oof.npz）。

总分 = 8 题期望分之和；对照 trial 标签算 CCC / RMSE / MAE、PHQ>=10 的 AUROC、逐题 MAE / QWK；
另附 Q-A 对数、每题被问题映射命中的 session 比例（检索质量），以及训练 MAE 基线（总是猜训练均值）。

  python scripts/analysis/compare_trial.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import N_ITEMS  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation.metrics import item_report, safe_auc, total_report  # noqa: E402
from src.labels import ITEM_COLS, load_labels  # noqa: E402
from src.preprocessing.pipeline import load_qa_pairs  # noqa: E402
from src.stage1.teacher import load_teacher  # noqa: E402

VARIANTS = ["two_track", "official", "compressed"]


def main():
    rows, items = [], {}
    base_cfg = load_config("config/trial/base.yaml")
    lab = load_labels(base_cfg).set_index("sid")
    edges = list(base_cfg.splits.phq_bin_edges)
    for v in VARIANTS:
        cfg = load_config(f"config/trial/{v}.yaml")
        p = ROOT / cfg.paths.outputs / cfg.experiment.name / "stage1" / "teacher_oof.npz"
        if not p.exists():
            print(f"[{v}] 没有 {p}，跳过")
            continue
        t = load_teacher(p)
        m = t["split"] == "train"
        sids, probs = t["sids"][m], t["probs"][m]
        y_items = lab.loc[sids, ITEM_COLS].to_numpy(float)
        y = lab.loc[sids, "phq_total"].to_numpy(float)
        yhat = (probs * np.arange(probs.shape[-1])).sum(-1).sum(-1)
        tr = total_report(y, yhat, edges)
        auc, ap = safe_auc(y >= 10, yhat)
        ir = item_report(y_items, probs)
        ret = json.load(open(ROOT / cfg.paths.outputs / cfg.experiment.name / "stage1" / "retrieval.json"))
        n_pairs = [len(load_qa_pairs(cfg, int(s))) for s in sids]
        mapped = {k: np.mean([ret[str(s)][str(k)]["max_r"] >= 0.999 for s in sids]) for k in range(1, N_ITEMS + 1)}
        rows.append({"variant": v, "n": tr["n"], "CCC": tr["ccc"], "RMSE": tr["rmse"], "MAE": tr["mae"], "AUROC": auc,
                     "item_MAE_mean": np.mean([ir[f"item{k}"]["mae"] for k in range(1, N_ITEMS + 1)]),
                     "item_QWK_mean": np.mean([ir[f"item{k}"]["qwk"] for k in range(1, N_ITEMS + 1)]),
                     "pairs_per_session": np.mean(n_pairs),
                     "items_mapped_frac": np.mean(list(mapped.values()))})
        items[v] = {k: ir[f"item{k}"]["mae"] for k in range(1, N_ITEMS + 1)}
        hist = [json.load(open(ROOT / cfg.paths.outputs / cfg.experiment.name / "stage1" / f"fold{f}" / "history.json"))
                for f in range(cfg.splits.n_folds)]
        rows[-1]["best_epochs"] = [h["best_epoch"] for h in hist]
    if not rows:
        return
    y_all = lab["phq_total"].to_numpy(float)
    print(f"基线（都猜均值 {y_all.mean():.2f}）MAE = {np.mean(np.abs(y_all - y_all.mean())):.2f}\n")
    df = pd.DataFrame(rows).set_index("variant")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df)
        print("\n逐题 MAE（期望分）:")
        print(pd.DataFrame(items).rename_axis("item"))
    out = ROOT / "outputs" / "trial_compare.json"
    json.dump({"summary": json.loads(df.to_json(orient="index")), "item_mae": items}, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
