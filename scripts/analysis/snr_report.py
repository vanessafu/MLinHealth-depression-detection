"""每个 session 的信噪比分析：底噪、受访者 / Ellie 的响度、Silero 语音概率分布。

参考标签（只用于分析，不进管线）：
  转写里 speaker == ELLIE，或句子和 Ellie 问题库模糊匹配 >= --ref-match -> Ellie；其余长度 >= 3 词的句子 -> 受访者
  python scripts/analysis/snr_report.py --sids 308 309 676
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from src.audio import load_wav  # noqa: E402
from src.preprocessing.ellie import normalize  # noqa: E402

SR, HOP = 16000, 512   # Silero 帧 = 32 ms


def frame_db(w):
    n = len(w) // HOP
    return 10 * np.log10((w[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)


def silero_probs(w):
    from silero_vad import load_silero_vad
    m = load_silero_vad()
    m.reset_states()
    x = torch.from_numpy(w)
    with torch.no_grad():
        return np.array([m(x[i:i + HOP], SR).item() for i in range(0, len(x) - HOP + 1, HOP)])


def ref_labels(sid, n, ref_match):
    bank = [normalize(t) for t in pd.read_csv(ROOT / "resources/ellie_questions.csv")["text"]]
    segs = json.load(open(ROOT / f"data/interim/transcripts/{sid}.json"))["segments"]
    lab = np.zeros(n, np.int8)            # 0 未标, 1 受访者, 2 Ellie
    for s in segs:
        t = normalize(s.get("text", ""))
        hit = process.extractOne(t, bank, scorer=fuzz.ratio) if t else None
        is_ellie = s.get("speaker") == "ELLIE" or (hit and hit[1] >= ref_match)
        if not is_ellie and len(t.split()) < 3:
            continue
        a, b = int(s["start"] * SR / HOP), int(s["end"] * SR / HOP)
        lab[a:b] = 2 if is_ellie else 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="*", type=int, default=[308, 309, 676])
    ap.add_argument("--ref-match", type=float, default=75)
    a = ap.parse_args()
    rows = []
    for sid in a.sids:
        w = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav")
        db, pr = frame_db(w), silero_probs(w)
        n = min(len(db), len(pr))
        db, pr = db[:n], pr[:n]
        lab = ref_labels(sid, n, a.ref_match)
        nz = db[db > -100]
        floor = np.percentile(nz, 10)
        P, E = lab == 1, lab == 2
        act = lambda m, db=db: np.percentile(db[m], 75)    # noqa: E731  有声帧响度
        r = {"sid": sid, "floor_db": floor,
             "part_db": act(P), "ellie_db": act(E),
             "snr_part": act(P) - floor, "snr_ellie": act(E) - floor, "ellie_minus_part": act(E) - act(P),
             "silero_p_part_med": np.median(pr[P]), "silero_p_ellie_med": np.median(pr[E]),
             "ellie_frames_p>0.5": (pr[E] > 0.5).mean(), "part_frames_p>0.5": (pr[P] > 0.5).mean(),
             "n_ellie_s": E.sum() * HOP / SR, "n_part_s": P.sum() * HOP / SR}
        # 最能分开两人的 Silero 阈值（只看语音帧）
        best = max(((t, ((pr[P] > t).mean() + (pr[E] <= t).mean()) / 2) for t in np.arange(0.05, 1.0, 0.05)),
                   key=lambda z: z[1])
        r["best_thr"], r["best_bal_acc"] = best
        # 只用能量分两人
        bestE = max(((t, ((db[P] > t).mean() + (db[E] <= t).mean()) / 2) for t in np.arange(-80, -10, 1.0)),
                    key=lambda z: z[1])
        r["best_energy_thr_db"], r["energy_bal_acc"] = bestE
        rows.append(r)
        np.savez(ROOT / f"data/interim/snr_{sid}.npz", db=db, pr=pr, lab=lab)
    df = pd.DataFrame(rows).set_index("sid").T
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(df.to_string())


if __name__ == "__main__":
    main()
