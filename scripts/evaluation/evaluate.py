"""融合（在 train OOF 上选 β）并计算全部指标，写到 outputs/<exp>/final/。

用法:
  python scripts/evaluation/evaluate.py                     # 完整管线（A7）：读 stage2/predictions.csv
  python scripts/evaluation/evaluate.py --text-only         # 只用文本，β = 0（A3）
  python scripts/evaluation/evaluate.py --text-only --set stage2.teacher.kind=zeroshot   # A1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import _bootstrap  # noqa: E402,F401
import pandas as pd  # noqa: E402

from src.config import base_argparser, config_from_args, exp_dir, resolve  # noqa: E402
from src.evaluation.report import build_final, evaluate, text_only_predictions  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.stage1.teacher import load_teacher  # noqa: E402


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    lab = load_labels(cfg)

    s1 = resolve(cfg.stage2.teacher.stage1_dir) if cfg.stage2.teacher.stage1_dir else exp_dir(cfg, "stage1")
    teacher = load_teacher(s1 / f"teacher_{cfg.stage2.teacher.kind}.npz")
    if args.text_only:
        pred = text_only_predictions(teacher)
        grid = [0.0]
        tag = f"text_{cfg.stage2.teacher.kind}"
    else:
        pred = pd.read_csv(exp_dir(cfg, "stage2") / "predictions.csv")
        grid = list(cfg.fusion.beta_grid)
        tag = "full"

    final, fusion_info = build_final(pred, lab, grid)
    metrics = {"fusion": fusion_info, **evaluate(final, lab, teacher, cfg)}
    out = exp_dir(cfg, "final")
    final.to_csv(out / f"predictions_{tag}.csv", index=False)
    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=1, default=float)

    print(f"beta = {fusion_info['beta']}")
    for split in ["train", "dev", "test"]:
        if split in metrics:
            m = metrics[split]["y_final"]
            print(f"[{split:5s}] CCC {m['ccc']:.3f}  RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  (n={m['n']})  "
                  f"mean-baseline RMSE {metrics[split]['mean_baseline_rmse']:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
