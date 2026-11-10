"""合并标签、做一致性检查（8 题之和 = PHQ_Score，PHQ_Binary = score≥10），生成并固定 5 折划分。

用法: python scripts/preprocessing/prepare_labels.py [--config config/default.yaml] [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import _bootstrap  # noqa: E402,F401

from src.config import base_argparser, config_from_args, resolve  # noqa: E402
from src.labels import consistency_report, load_labels  # noqa: E402
from src.splits import folds_path, make_folds  # noqa: E402


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 folds.csv（会改变两个阶段共用的划分！）")
    args = ap.parse_args()
    cfg = config_from_args(args)

    lab = load_labels(cfg)
    out_dir = resolve(cfg.paths.splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out_dir / "labels_merged.csv", index=False)
    rep = consistency_report(lab)
    print(json.dumps(rep, indent=1, default=str))

    fp = folds_path(cfg)
    if fp.exists() and not args.force:
        print(f"{fp} 已存在，保持不变（用 --force 重新生成）")
        return
    folds = make_folds(lab, cfg)
    folds.to_csv(fp, index=False)
    print(f"分层变量: {folds.attrs.get('stratify_cols_used')}  ->  {fp}")
    print(folds.groupby("fold").agg(n=("sid", "size"), phq_mean=("phq_total", "mean")).to_string())


if __name__ == "__main__":
    main()
