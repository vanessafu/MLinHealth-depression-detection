"""Stage 2：5 折 × 多个 seed 训练音频残差模型。

用法:
  python scripts/stage2-audio/train_stage2.py --grid    # 先搜 w_t × w_m × λ（27 组），按 CV 总分 CCC 选最优
  python scripts/stage2-audio/train_stage2.py           # 用配置里的（或 --grid 选出的）超参数跑最终模型并保存预测
  python scripts/stage2-audio/train_stage2.py --use-best-grid   # 自动读取 grid 结果里的最优组合
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import base_argparser, config_from_args, exp_dir, resolve, save_config  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.sessions import available_sids  # noqa: E402
from src.splits import load_folds  # noqa: E402
from src.stage2.dataset import Bags  # noqa: E402
from src.stage2.train import grid_configs, predictions_frame, run_cv, save_json  # noqa: E402


def teacher_path(cfg) -> Path:
    s1 = resolve(cfg.stage2.teacher.stage1_dir) if cfg.stage2.teacher.stage1_dir else exp_dir(cfg, "stage1")
    return s1 / f"teacher_{cfg.stage2.teacher.kind}.npz"


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--use-best-grid", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = cfg.experiment.device
    s2 = exp_dir(cfg, "stage2")

    lab = load_labels(cfg)
    folds = load_folds(cfg)
    folds = folds[folds.sid.isin(available_sids(cfg, lab, "train", need=("wavlm",)))].reset_index(drop=True)
    eval_sids = {s: available_sids(cfg, lab, s, need=("wavlm",)) for s in ["dev", "test"]}
    eval_sids = {k: v for k, v in eval_sids.items() if v}
    all_sids = folds.sid.tolist() + sum(eval_sids.values(), [])
    tp = teacher_path(cfg)
    print("teacher:", tp)
    bags = Bags(cfg, all_sids, tp, lab)
    sid_to_i = {s: i for i, s in enumerate(bags.sids)}
    eval_idx = {s: [sid_to_i[x] for x in v] for s, v in eval_sids.items()}

    if args.grid:
        rows = []
        for params, c in grid_configs(cfg):
            res = run_cv(c, bags, folds, {}, device)
            rows.append({**params, "cv_ccc": res["cv_ccc"]})
            print(json.dumps(rows[-1]))
        df = pd.DataFrame(rows).sort_values("cv_ccc", ascending=False)
        df.to_csv(s2 / "grid.csv", index=False)
        print(df.head(10).to_string(index=False))
        return

    if args.use_best_grid:
        best = pd.read_csv(s2 / "grid.csv").iloc[0]
        for k in ["w_tired", "w_mov", "lambda_res"]:
            cfg.stage2.loss[k] = float(best[k])
        print("using best grid:", {k: cfg.stage2.loss[k] for k in ["w_tired", "w_mov", "lambda_res"]})

    save_config(cfg, s2)
    res = run_cv(cfg, bags, folds, eval_idx, device, save_dir=s2)
    frames = [predictions_frame(bags, res["train_idx"], res["oof"], "train")]
    for s, idx in eval_idx.items():
        frames.append(predictions_frame(bags, idx, res["eval"][s], s))
    pd.concat(frames).to_csv(s2 / "predictions.csv", index=False)
    np.savez(s2 / "attention.npz", **{f"{s}_{bags.sids[i]}": a for s in eval_idx
                                      for i, a in zip(eval_idx[s], res["eval_attn"][s])})
    save_json({"cv_ccc": res["cv_ccc"], "runs": res["log"]}, s2 / "cv_summary.json")
    print(f"CV CCC (train OOF, y_tot) = {res['cv_ccc']:.4f}  ->  {s2 / 'predictions.csv'}")


if __name__ == "__main__":
    main()
