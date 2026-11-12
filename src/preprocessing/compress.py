"""不降噪的逐帧动态范围压缩：轻的说话人抬到和响的差不多，响的不动，时间轴不变。

为什么不降噪：309 里 Ellie 原始响度 ~-69 dB，和底噪一样；noisereduce 一过就到 -105 dB（被当噪声抹掉）。
但 Whisper 在原始音频里其实听得见她——只是同一个解码窗口里有响的受访者时，会把她当背景跳过。
把她抬上来后无 VAD 解码，309 开场白 / 300 PHQ 问答都几乎逐字还原。
门限：比底噪高不到 gate_db 的帧不放大，否则 Whisper 会在放大的底噪上循环幻觉（308："I'm from New York." x10）。
"""
from __future__ import annotations

import numpy as np

from .enhance import _frame_db

HOP_S = 0.05


def _win_max(x: np.ndarray, r: int) -> np.ndarray:
    """每帧 ±r 帧内的最大值：一个词内部的局部响度保持稳定，不会被词间小停顿拉低。"""
    p = np.pad(x, r, mode="edge")
    return np.max(np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1), axis=1)


def compress(wav: np.ndarray, sr: int, *, target_db: float = -24.0, max_gain_db: float = 30.0,
             gate_db: float | None = 3.0, win_s: float = 0.35, smooth_s: float = 0.25) -> tuple[np.ndarray, dict]:
    """增益 = clip(target - 局部响度, 0, max_gain)（dB），门限以下不放大，dB 上平滑后插值到采样点，tanh 软限幅。"""
    hop = int(HOP_S * sr)
    fd = _frame_db(wav, hop)
    local = _win_max(fd, max(1, int(win_s / HOP_S / 2)))
    g = np.clip(target_db - local, 0.0, max_gain_db)
    floor = float(np.percentile(fd[fd > -100], 10)) if (fd > -100).any() else -100.0
    if gate_db is not None:
        g = np.where(local > floor + gate_db, g, 0.0)
    k = max(1, int(smooth_s / HOP_S))
    g = np.convolve(g, np.ones(k) / k, mode="same")
    gs = np.interp(np.arange(len(wav)), (np.arange(len(g)) + 0.5) * hop, g)
    out = np.tanh(wav * 10 ** (gs / 20) / 0.95) * 0.95
    info = {"floor_db": floor, "gain_db_p10_p50_p90": [float(v) for v in np.percentile(g, [10, 50, 90])]}
    return out.astype(np.float32), info
