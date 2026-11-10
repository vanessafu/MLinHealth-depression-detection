"""Stage 2 训练前的两个前提检查（第 6 节）：
  1. tired 3 级、moving 二分类的标签数量
  2. 音频特征能否解释 OOF 残差 y - y_text（session 级特征 + 岭回归，5 折 CV）

用法:
  python scripts/stage2-audio/precheck.py                    # 用 WavLM 缓存的层平均 + 片段平均作 session 特征
  python scripts/stage2-audio/precheck.py --features egemaps # 用 paths.egemaps_dir 下的官方 eGeMAPS
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import base_argparser, config_from_args, exp_dir, resolve  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.preprocessing import feature_path  # noqa: E402
from src.splits import load_folds  # noqa: E402
from src.stage1.teacher import load_teacher, teacher_features  # noqa: E402


def session_feats(cfg, sids, kind):
    X = []
    for s in sids:
        if kind == "wavlm":
            f = torch.load(feature_path(cfg, s))["feats"].float()       # [M, L, D]
            X.append(f.mean(dim=(0, 1)).numpy())
        else:
            p = next(resolve(cfg.paths.egemaps_dir).glob(f"{s}_*egemaps*.csv"))
            df = pd.read_csv(p, sep=None, engine="python").select_dtypes("number")
            X.append(np.concatenate([df.mean().to_numpy(), df.std().to_numpy()]))
    return np.nan_to_num(np.stack(X))


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--features", choices=["wavlm", "egemaps"], default="wavlm")
    args = ap.parse_args()
    cfg = config_from_args(args)
    lab = load_labels(cfg)
    tr = lab[lab.split == "train"]

    print("== 1. 标签数量（train）")
    print("tired3:", tr.tired3.value_counts().sort_index().to_dict())
    n_pos = int((tr.mov == 1).sum())
    print("moving>=1:", tr.mov.value_counts().sort_index().to_dict())
    if n_pos < 25:
        print(f"  !! moving 正样本只有 {n_pos} 个（<25），建议把 stage2.loss.w_mov 固定为 0")

    print(f"\n== 2. 音频能否解释 OOF 残差（features={args.features}）")
    s1 = Path(cfg.stage2.teacher.stage1_dir) if cfg.stage2.teacher.stage1_dir else exp_dir(cfg, "stage1")
    t = load_teacher(resolve(s1) / "teacher_oof.npz")
    m = t["split"] == "train"
    y_text = pd.Series(teacher_features(t["probs"][m])["y_text"], index=t["sids"][m])
    folds = load_folds(cfg)
    folds = folds[folds.sid.isin(y_text.index)]
    sids = folds.sid.tolist()
    y = lab.set_index("sid").loc[sids, "phq_total"].to_numpy()
    resid = y - y_text.loc[sids].to_numpy()
    X = session_feats(cfg, sids, args.features)
    pred = np.zeros_like(resid)
    for f in sorted(folds.fold.unique()):
        te = (folds.fold == f).to_numpy()
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X[~te], resid[~te])
        pred[te] = mdl.predict(X[te])
    r, p = pearsonr(resid, pred)
    print(f"CV 相关系数 r = {r:.3f} (p = {p:.3g})，残差 std = {resid.std():.2f}")
    if r < 0.1:
        print("  !! 相关接近 0：音频很可能解释不了文本的残差，建议先改进 Stage 1")


if __name__ == "__main__":
    main()
