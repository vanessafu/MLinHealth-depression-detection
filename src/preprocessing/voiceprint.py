"""逐句判别 Ellie / 受访者：声纹为主（Ellie 是合成语音，音色和真人差得多，不受两人谁响谁轻影响）。

  声纹    pyannote wespeaker ResNet34 x-vector。5 个 session 的锚点留一 5-NN 准确率 0.83–0.97
          （WavLM-SV 只有 0.46–0.92）；单元太短就把自身重复补到 1 s，不往两边扩（免得混进另一个人）
  锚点    和 Ellie 台词库相似度 >= anchor_e 且 >= 3 个词 = Ellie；< anchor_p 且 >= 5 个词 = 受访者
  分类    z = voice_weight * 可靠度(时长) * kNN 投票（按类别加权）+ 台词线索（>= 3 个词、只加不减）+ 响度线索，
          自训练几轮
          kNN 的邻居只用 >= 1 s 的已标单元；可靠度 = clip(时长 / 1.5 s, 0.3, 1)："okay" 这种主要靠响度
5 个 session 上验证（实验脚本已删）：受访者被错标成 Ellie 的词 0–5%（按官方受访者转写算）。
"""
from __future__ import annotations

import numpy as np

from ..config import Cfg
from .asr_merge import word_iv

_CACHE: dict = {}


def _embedder(model: str, device: str):
    if "emb" not in _CACHE:
        import torch
        from pyannote.audio import Inference, Model

        _CACHE["emb"] = Inference(Model.from_pretrained(model), window="whole", device=torch.device(device))
    return _CACHE["emb"]


def embed(audio: np.ndarray, ivs: list[tuple[float, float]], sr: int, model: str, device: str,
          min_sec: float = 1.0) -> np.ndarray:
    import torch

    x = np.concatenate([audio[int(a * sr): int(b * sr)] for a, b in ivs]) if ivs else np.zeros(0, np.float32)
    if len(x) < 0.05 * sr and ivs:
        # 词时间长度为 0 / 落在音频末尾之外（473）：两边各扩 0.1 s 再切
        a, b = min(i[0] for i in ivs) - 0.1, max(i[1] for i in ivs) + 0.1
        x = audio[max(0, int(a * sr)): max(0, int(b * sr))]
    if len(x) == 0:
        x = np.full(sr, 1e-4, np.float32)                 # 还是切不到：给一段近静音（这种单元可靠度最低）
    if len(x) < min_sec * sr:
        x = np.tile(x, int(np.ceil(min_sec * sr / max(len(x), 1))))[: int(min_sec * sr)]
    x = (x / (np.abs(x).max() + 1e-6) * 0.9).astype(np.float32)
    e = np.asarray(_embedder(model, device)({"waveform": torch.from_numpy(x)[None], "sample_rate": sr})).ravel()
    return e / (np.linalg.norm(e) + 1e-9)


def classify(emb: np.ndarray, txt: np.ndarray, n_words: np.ndarray, dur: np.ndarray, lvl: np.ndarray, vc,
             iters: int = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (是否 Ellie, z, info)。vc: preprocessing.transcribe.speaker 配置。"""
    n = len(txt)
    e_anchor = (txt >= vc["anchor_e"]) & (n_words >= 3)
    p_anchor = (txt < vc["anchor_p"]) & (n_words >= 5)
    info = {"n_anchor_E": int(e_anchor.sum()), "n_anchor_P": int(p_anchor.sum())}
    # 台词线索只当正证据：像台词 -> Ellie；不像不代表是受访者（Ellie 也有库外的话），所以 < 0.7 不扣分
    z_txt = np.where(n_words >= 3, vc["text_weight"] * np.maximum(txt - 0.7, 0.0) / 0.15, 0.0)
    if not e_anchor.any() or not p_anchor.any():          # 锚点不够：只靠台词
        info["fallback"] = "text_only"
        return e_anchor | (z_txt > 0), z_txt, info

    lab = np.where(e_anchor, 1.0, np.where(p_anchor, 0.0, np.nan))
    S = emb @ emb.T
    np.fill_diagonal(S, -9)
    rel = np.clip(dur / 1.5, 0.3, 1.0)
    long_ = dur >= 1.0
    z = z_txt
    for _ in range(iters):
        known = np.flatnonzero(~np.isnan(lab) & long_)
        # 类别加权投票：每票除以该类已标单元数，Ellie 锚点比受访者少很多时（309：15 vs 27）不被多数类淹没
        nE, nP = max((lab[known] == 1).sum(), 1), max((lab[known] == 0).sum(), 1)
        v = np.zeros(n)
        for i in range(n):
            cand = known[known != i]
            nn = cand[np.argsort(-S[i, cand])[: vc["knn"]]]
            if len(nn):
                e, p = (lab[nn] == 1).sum() / nE, (lab[nn] == 0).sum() / nP
                v[i] = (e - p) / (e + p)
        lE, lP = np.median(lvl[lab == 1]), np.median(lvl[lab == 0])
        sep = abs(lE - lP)
        z_lvl = np.clip((np.abs(lvl - lP) - np.abs(lvl - lE)) / max(sep, 1.0), -1, 1) * min(sep / 6, 2.0)
        z = vc["voice_weight"] * rel * v + z_txt + z_lvl
        new = (z > 0).astype(float)
        new[e_anchor] = 1.0
        if np.array_equal(new, lab):
            break
        lab = new
    cE, cP = emb[lab == 1].mean(0), emb[lab == 0].mean(0)
    info.update({"cos_E_P": float(cE @ cP / np.linalg.norm(cE) / np.linalg.norm(cP)),
                 "level_E_db": float(lE), "level_P_db": float(lP)})
    return lab == 1, z, info


def assign_speakers_voiceprint(units: list[dict], wav: np.ndarray, emb_audio: np.ndarray, cfg: Cfg, device: str) -> dict:
    """给每个单元写 speaker / p_ellie / level_db（以及每个词的 speaker）。units 需要先有 text_score（speaker.text_scores）。
    和 speaker.assign_speakers（two_track 用）是两套判别。"""
    from .speaker import segment_levels

    if not units:
        return {}
    sr = cfg.preprocessing.sample_rate
    vc = cfg.preprocessing.transcribe.speaker
    lvl = segment_levels(wav, units, sr)
    emb = np.stack([embed(emb_audio, word_iv(u), sr, vc.model, device) for u in units])
    txt = np.array([u["text_score"] for u in units])
    nw = np.array([len(u["words"]) for u in units])
    dur = np.array([sum(b - a for a, b in word_iv(u)) for u in units])
    is_e, z, info = classify(emb, txt, nw, dur, lvl, vc)
    for u, e, zz, lv in zip(units, is_e, z, lvl):
        u["speaker"] = "ELLIE" if e else "PARTICIPANT"
        u["p_ellie"], u["level_db"] = float(1 / (1 + np.exp(-zz))), float(lv)
        for w in u["words"]:
            w["speaker"] = u["speaker"]
    return info
