"""Ellie 轨：录音里 Ellie 比受访者小 20–30 dB、只比底噪高几 dB，Silero 和 pyannote 都抓不到她。

做法（分轨）：
  1) 平稳底噪抑制（noisereduce stationary，谱门限 / 类维纳滤波），把 Ellie 的信噪比提高约 15 dB
  2) 原始音频上的 Silero VAD 区间基本都是受访者 -> 把这些区间（+pad）挖掉，剩下的就是 Ellie 轨
  3) Ellie 轨里每段间隙单独做 RMS 标准化（只看有声帧），把 Ellie 拉到和受访者一样响；只有底噪的间隙保持静音
时间轴不变，所以 Ellie 轨上转写出的句子就标为 ELLIE，不需要说话人分离。
"""
from __future__ import annotations

import numpy as np


def _db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(x, 1e-12))


def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return _db((x[: n * hop].reshape(n, hop) ** 2).mean(1)) if n else np.array([-120.0])


def denoise(wav: np.ndarray, sampling_rate: int, *, prop_decrease: float = 1.0, n_std: float = 1.5) -> np.ndarray:
    import noisereduce as nr

    return nr.reduce_noise(y=wav, sr=sampling_rate, stationary=True, prop_decrease=prop_decrease,
                           n_std_thresh_stationary=n_std).astype(np.float32)


def ellie_track(wav: np.ndarray, sr: int, participant: list[tuple[float, float]], *, pad_s: float = 0.3,
                target_db: float = -24.0, max_gain_db: float = 50.0, active_pct: float = 95.0,
                min_snr_db: float = 8.0, fade_ms: float = 30.0, prop_decrease: float = 1.0,
                n_std: float = 1.5) -> tuple[np.ndarray, dict]:
    """受访者区间（+pad）置零，其余间隙降噪后逐段 RMS 标准化。返回 (ellie_wav, info)。"""
    x = denoise(wav, sr, prop_decrease=prop_decrease, n_std=n_std)
    keep = np.ones(len(x), bool)
    for s, e in participant:
        keep[max(0, int((s - pad_s) * sr)): int((e + pad_s) * sr)] = False
    hop = int(0.05 * sr)                                   # 50 ms 帧
    fd = _frame_db(x[keep], hop)
    fd = fd[fd > -100]                                     # 降噪后会有纯零帧，不算进底噪
    floor_db = float(np.percentile(fd, 20)) if len(fd) else -100.0
    out = np.zeros_like(x)
    fade = int(fade_ms / 1000 * sr)
    edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
    gains = []
    for a, b in zip(edges[::2], edges[1::2]):
        seg = x[a:b]
        if len(seg) < hop * 4:
            continue
        active_db = float(np.percentile(_frame_db(seg, hop), active_pct))   # 这段里"有声"部分的响度
        if active_db - floor_db < min_snr_db:              # 只有底噪：不放大，防止 Whisper 幻觉
            continue
        g_db = min(max(target_db - active_db, 0.0), max_gain_db)
        env = np.ones(len(seg), np.float32)
        f = min(fade, len(seg) // 2)
        if f:
            env[:f] = np.linspace(0, 1, f)
            env[-f:] = np.linspace(1, 0, f)
        out[a:b] = np.tanh(seg * 10 ** (g_db / 20) * env / 0.95) * 0.95   # 软限幅
        gains.append(g_db)
    info = {"floor_db": floor_db, "boosted_gaps": len(gains),
            "median_gain_db": float(np.median(gains)) if gains else 0.0}
    return out.astype(np.float32), info
