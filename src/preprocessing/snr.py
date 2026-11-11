"""Step 1（诊断）: 每个 session 的信噪比分析，写进 vad/<sid>.json 的 "snr" 字段，用于 QC。

录音之间差别很大（同一个配置不可能都合适）：
  309  底噪 -72，受访者 -30，Ellie -60（Ellie 极小声）
  676  底噪 -75，受访者 -48，Ellie -64（整体很小声，Ellie 小 16 dB）
  308  底噪 -41，受访者 -31，Ellie -22（底噪高，Ellie 反而更响）

在 "有声帧"（0.5 s 平滑能量 > 底噪 + active_above_floor_db）上对能量做 Otsu 二分 -> 响 / 轻两簇的响度、差值、
分离度 d'；对 Silero 概率也做 Otsu（silero_otsu）。
试过用 silero_otsu 当每个 session 的 VAD 阈值：3 个 session 上不如 "低 + 高两个阈值融合"（见 transcribe.py），
同样 SNR 的 309 / 676 最优阈值正好相反，所以 VAD 不再按 SNR 设阈值；说话人判别里的响度权重按 d' 自适应（speaker.py）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..config import Cfg

HOP = 512   # Silero 在 16 kHz 下的帧长（32 ms）


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    h, e = np.histogram(x, bins=bins)
    c, w = (e[:-1] + e[1:]) / 2, h / max(h.sum(), 1)
    best, thr = -1.0, float(np.median(x))
    for i in range(1, bins):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (w[:i] * c[:i]).sum() / w0, (w[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(e[i])
    return thr


def frame_db(wav: np.ndarray) -> np.ndarray:
    n = len(wav) // HOP
    return 10 * np.log10((wav[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)


def silero_probs(wav: np.ndarray, sr: int) -> np.ndarray:
    import torch

    from .vad import get_model

    m = get_model()
    m.reset_states()
    x = torch.from_numpy(wav)
    with torch.no_grad():
        p = [m(x[i:i + HOP], sr).item() for i in range(0, len(x) - HOP + 1, HOP)]
    m.reset_states()
    return np.asarray(p)


def analyze(wav: np.ndarray, cfg: Cfg) -> dict:
    ac = cfg.preprocessing.snr
    sr = cfg.preprocessing.sample_rate
    db, pr = frame_db(wav), silero_probs(wav, sr)
    n = min(len(db), len(pr))
    db, pr = db[:n], pr[:n]
    win = max(1, int(ac.smooth_s * sr / HOP))
    s_db = 10 * np.log10(uniform_filter1d(10 ** (db / 10), win) + 1e-12)
    s_pr = uniform_filter1d(pr, win)
    nz = db[db > -100]
    floor = float(np.percentile(nz, 10)) if len(nz) else -100.0
    act = s_db > floor + ac.active_above_floor_db
    if act.sum() < 50:                                  # 几乎没声音：用默认
        return {"floor_db": floor, "fallback": True}
    p_thr = _otsu(s_pr[act])
    e_thr = _otsu(s_db[act])
    hi, lo = s_db[act & (s_db > e_thr)], s_db[act & (s_db <= e_thr)]
    hi_db, lo_db = float(np.median(hi)), float(np.median(lo))
    sd = float(np.sqrt((hi.var() + lo.var()) / 2)) + 1e-6
    return {
        "floor_db": floor, "active_frac": float(act.mean()),
        "loud_db": hi_db, "quiet_db": lo_db, "loud_minus_quiet_db": hi_db - lo_db,
        "snr_loud_db": hi_db - floor, "snr_quiet_db": lo_db - floor,
        "separation_dprime": (hi_db - lo_db) / sd, "energy_split_db": e_thr,
        "silero_otsu": p_thr,
    }
