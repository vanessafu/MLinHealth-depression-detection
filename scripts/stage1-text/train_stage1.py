"""Stage 1：K 折训练 LoRA 并生成 OOF 老师预测。

用法:
  python scripts/stage1-text/train_stage1.py                    # 跑全部 5 折，然后汇总成 teacher_oof.npz
  python scripts/stage1-text/train_stage1.py --folds 0 1        # 只跑部分折（可以分到多张卡 / 多台机器）
  python scripts/stage1-text/train_stage1.py --assemble-only    # 所有折跑完后只做汇总
  python scripts/stage1-text/train_stage1.py --zero-shot        # 基线 S1-B0 / A1
  python scripts/stage1-text/train_stage1.py --full-fit         # 消融 A8：非 OOF 老师（需先跑完 5 折）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from src.config import base_argparser, config_from_args, exp_dir, save_config  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.sessions import available_sids  # noqa: E402
from src.splits import fold_ids, load_folds  # noqa: E402
from src.stage1.retrieval import load_retrieval  # noqa: E402
from src.stage1.train import assemble_oof, run_fold, run_full_fit, run_zero_shot  # noqa: E402


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--full-fit", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="--full-fit 的 epoch 数；默认取各折最佳 epoch 的中位数")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = cfg.experiment.device

    s1 = exp_dir(cfg, "stage1")
    save_config(cfg, s1)
    lab = load_labels(cfg)
    ret = load_retrieval(s1 / "retrieval.json")
    folds = load_folds(cfg)
    have = set(available_sids(cfg, lab, "train"))
    folds = folds[folds.sid.isin(have)]
    eval_sids = {s: available_sids(cfg, lab, s) for s in ["dev", "test"]}
    eval_sids = {k: v for k, v in eval_sids.items() if v}
    n_folds = cfg.splits.n_folds

    if args.zero_shot:
        print(run_zero_shot(cfg, {"train": folds.sid.tolist(), **eval_sids}, ret, s1, device))
        return
    if args.full_fit:
        n_ep = args.epochs
        if n_ep is None:
            best = [json.load(open(s1 / f"fold{f}" / "history.json"))["best_epoch"] for f in range(n_folds)]
            n_ep = int(np.median(best)) + 1
        print(run_full_fit(cfg, folds.sid.tolist(), eval_sids, ret, lab, s1, device, n_ep))
        return
    if not args.assemble_only:
        for f in (args.folds if args.folds is not None else range(n_folds)):
            tr, va = fold_ids(folds, f)
            print(f"===== fold {f}: train {len(tr)}  val {len(va)}")
            run_fold(cfg, f, tr, va, eval_sids, ret, lab, s1, device)
    if all((s1 / f"fold{f}" / "probs.npz").exists() for f in range(n_folds)):
        print("teacher ->", assemble_oof(s1, n_folds, list(eval_sids)))
    else:
        print("还有折没跑完，暂不汇总 teacher_oof.npz")


if __name__ == "__main__":
    main()
