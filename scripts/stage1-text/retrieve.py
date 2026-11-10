"""为每个 session × 每道题检索候选证据（问题映射表 + bge），写到 outputs/<exp>/stage1/retrieval.json。

用法:
  python scripts/stage1-text/retrieve.py
  python scripts/stage1-text/retrieve.py --set experiment.name=A4_no_map stage1.retrieval.use_question_map=false
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from src import N_ITEMS  # noqa: E402
from src.config import base_argparser, config_from_args, exp_dir, save_config  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.sessions import available_sids  # noqa: E402
from src.stage1.retrieval import build_retrieval, save_retrieval  # noqa: E402


def main():
    args = base_argparser(__doc__).parse_args()
    cfg = config_from_args(args)
    lab = load_labels(cfg)
    sids = sum((available_sids(cfg, lab, s) for s in ["train", "dev", "test"]), [])
    ret = build_retrieval(cfg, sids, cfg.experiment.device)
    out = exp_dir(cfg, "stage1")
    save_config(cfg, out)
    save_retrieval(ret, out / "retrieval.json")

    # 用于校准 evidence_threshold：看各题 max_r 的分布
    print("各题 max_j r_ijk 分位数 (10/50/90%) 与 ev=1 比例:")
    for k in range(1, N_ITEMS + 1):
        mr = np.array([ret[s][k]["max_r"] for s in sids])
        ev = np.mean([ret[s][k]["ev"] for s in sids])
        print(f"  item{k}: {np.percentile(mr, [10, 50, 90]).round(3)}  ev={ev:.2f}")
    print(f"-> {out / 'retrieval.json'}")


if __name__ == "__main__":
    main()
