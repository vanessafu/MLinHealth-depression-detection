"""画 session 的语谱图：上 = 整场，下 = 从访谈开始起的一段放大；下方叠加 VAD 区间和逐句说话人。

  python scripts/analysis/plot_spectrogram.py --sids 308 309 [--zoom 60] [--start 40]
输出: outputs/plots/{sid}_spectrogram.png
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 登录节点 256 核，线程互抢很慢
    os.environ.setdefault(_v, "4")
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import find_audio, load_wav  # noqa: E402
from src.config import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sids", nargs="+", type=int, default=[308, 309])
ap.add_argument("--zoom", type=float, default=60.0, help="放大窗口长度（秒）")
ap.add_argument("--start", type=float, default=None, help="放大窗口起点；默认 = 访谈开始")
a = ap.parse_args()
cfg = load_config()
sr = cfg.preprocessing.sample_rate
audio = find_audio(cfg)
out = ROOT / "outputs/plots"
out.mkdir(parents=True, exist_ok=True)
COL = {"ELLIE": "#d62728", "PARTICIPANT": "#1f77b4"}


def spec_db(x, nperseg, max_cols=2400, max_rows=400):
    """语谱 (dB)。先在功率域按块平均压到屏幕分辨率，否则整场 ~1400 万格画起来要几分钟。"""
    f, t, S = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, scaling="spectrum")
    ct, cf = max(1, len(t) // max_cols), max(1, len(f) // max_rows)
    nt, nf = len(t) // ct, len(f) // cf
    S = S[: nf * cf, : nt * ct].reshape(nf, cf, nt, ct).mean((1, 3))
    return f[: nf * cf : cf], t[: nt * ct : ct], 10 * np.log10(S + 1e-12)


def lanes(ax, t0, t1, vad, segs):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 2)
    for s, e in vad:
        if e > t0 and s < t1:
            ax.axvspan(s, e, ymin=0.55, ymax=0.95, color="0.4")
    for sg in segs:
        if sg["end"] > t0 and sg["start"] < t1:
            ax.axvspan(sg["start"], sg["end"], ymin=0.05, ymax=0.45, color=COL.get(sg.get("speaker"), "0.7"))
    ax.set_yticks([0.5, 1.5], ["speaker", "VAD"])
    ax.tick_params(axis="y", length=0)


for sid in a.sids:
    x = load_wav(audio[sid], sr)
    dur = len(x) / sr
    vad_j = json.load(open(ROOT / f"data/interim/vad/{sid}.json"))
    thr = sorted(vad_j["speech_by_threshold"], key=float)[0]
    vad = vad_j["speech_by_threshold"][thr]
    tr = json.load(open(ROOT / f"data/interim/transcripts/{sid}.json"))
    segs = tr["segments"]
    z0 = a.start if a.start is not None else max(0.0, tr.get("interview_start", 0.0) - 2)
    z1 = min(dur, z0 + a.zoom)
    snr = vad_j.get("snr", {})

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 0.45, 3, 0.45])
    vmax = None
    for row, (t0, t1, nps) in enumerate([(0, dur, 2048), (z0, z1, 512)]):
        seg = x[int(t0 * sr): int(t1 * sr)]
        f, t, S = spec_db(seg, nps)
        vmax = vmax if vmax is not None else np.percentile(S, 99.5)
        ax = fig.add_subplot(gs[2 * row])
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
                       extent=(t0, t0 + len(seg) / sr, 0, f[-1] / 1000), interpolation="nearest")
        ax.set_ylabel("kHz")
        ax.set_xlim(t0, t1)
        ax.set_xticklabels([])
        if row == 0:
            ax.axvspan(z0, z1, color="cyan", alpha=0.12)
            g = {k: snr.get(k, float("nan")) for k in ("floor_db", "loud_db", "quiet_db", "separation_dprime")}
            ax.set_title(f"{sid}: whole session ({dur / 60:.1f} min) | floor {g['floor_db']:.0f} dB, "
                         f"loud {g['loud_db']:.0f} dB, quiet {g['quiet_db']:.0f} dB, d'={g['separation_dprime']:.1f}"
                         f" | interview starts {tr.get('interview_start', 0):.1f}s", loc="left")
        else:
            ax.set_title(f"zoom {t0:.0f}-{t1:.0f} s", loc="left")
            for sg in segs:   # 放大图上标出每句的开头几个词
                if t0 <= sg["start"] < t1:
                    ax.text(sg["start"], 7.6, sg["text"].strip()[:28], color=COL.get(sg.get("speaker"), "w"),
                            fontsize=7, rotation=0, va="top", clip_on=True,
                            bbox=dict(facecolor="white", alpha=0.6, lw=0, pad=0.5))
        fig.colorbar(im, ax=ax, label="dB", pad=0.005)
        la = fig.add_subplot(gs[2 * row + 1])
        lanes(la, t0, t1, vad, segs)
        la.set_xlabel("time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COL["ELLIE"], COL["PARTICIPANT"], "0.4")]
    fig.legend(handles, ["ELLIE", "PARTICIPANT", f"VAD speech (thr {thr})"], loc="upper right", ncol=3)
    p = out / f"{sid}_spectrogram.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)
