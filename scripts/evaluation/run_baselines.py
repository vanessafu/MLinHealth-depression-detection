"""基线：A0（预测训练集均值）和 A1（Stage 1 零样本，若已生成 teacher_zeroshot.npz）。

用法（推荐配合 config/eval_baseline.yaml）:
  python scripts/stage1-text/retrieve.py       --config config/eval_baseline.yaml
  python scripts/stage1-text/train_stage1.py   --config config/eval_baseline.yaml --zero-shot
  python scripts/evaluation/run_baselines.py   --config config/eval_baseline.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from src.config import base_argparser, config_from_args, exp_dir  # noqa: E402
from src.evaluation.metrics import total_report  # noqa: E402
from src.evaluation.report import build_final, evaluate, text_only_predictions  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.stage1.teacher import load_teacher  # noqa: E402


def main():
    cfg = config_from_args(base_argparser(__doc__).parse_args())
    lab = load_labels(cfg)
    edges = cfg.evaluation.severity_bin_edges
    out = exp_dir(cfg, "final")
    res = {}

    # A0: 训练集均值
    mean = float(lab.loc[lab.split == "train", "phq_total"].mean())
    res["A0_mean"] = {"train_mean": mean}
    for split in ["train", "dev", "test"]:
        y = lab.loc[(lab.split == split) & lab.phq_total.notna(), "phq_total"].to_numpy()
        if len(y):
            res["A0_mean"][split] = total_report(y, np.full(len(y), mean), edges)

    # A1: 零样本
    zs = exp_dir(cfg, "stage1") / "teacher_zeroshot.npz"
    if zs.exists():
        t = load_teacher(zs)
        final, _ = build_final(text_only_predictions(t), lab, [0.0])
        res["A1_zeroshot"] = evaluate(final, lab, t, cfg)
    else:
        print(f"未找到 {zs}，跳过 A1（先跑 train_stage1.py --zero-shot）")

    with open(out / "baselines.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for name, r in res.items():
        for split in ["dev", "test"]:
            m = r.get(split, {})
            m = m.get("y_final", m)
            if "ccc" in m:
                print(f"{name:12s} [{split}] CCC {m['ccc']:.3f}  RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}")
    print(f"-> {out / 'baselines.json'}")


if __name__ == "__main__":
    main()
