"""试跑用的标签：只取官方 dev 集里 data/raw 下有音频的 session，改写成 data/trial/labels/train_split.csv，
这样 Stage 1 的 5 折（OOF）就在这些 session 上跑。不碰官方的 data/labels 和 data/splits。

  python scripts/analysis/trial_setup.py            # 默认 split=dev
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import find_audio  # noqa: E402
from src.config import load_config, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="官方哪个 split 当试跑训练池")
    ap.add_argument("--out", default="data/trial/labels")
    a = ap.parse_args()

    cfg = load_config()
    src = resolve(cfg.paths.labels_dir)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    have = set(find_audio(cfg))
    df = pd.read_csv(src / cfg.labels.split_files[a.split])
    det = pd.read_csv(src / cfg.labels.detailed_file)
    df = df[df[cfg.labels.id_col].isin(have) & df[cfg.labels.id_col].isin(det[cfg.labels.id_col])]
    df.to_csv(out / cfg.labels.split_files["train"], index=False)
    shutil.copy(src / cfg.labels.detailed_file, out / cfg.labels.detailed_file)
    print(f"{len(df)} sessions ({a.split}, 有音频且有逐题标签) -> {out}")
    print(" ".join(str(s) for s in sorted(df[cfg.labels.id_col])))


if __name__ == "__main__":
    main()
