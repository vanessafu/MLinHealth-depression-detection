"""Step 3b: 逐句判定说话人（Ellie / 受访者），三个线索加权，权重随本 session 的信噪比自适应。

线索（都转成 "是 Ellie" 的对数几率 logit）：
  question / first_person  问句 +；第一人称词比例高 -（受访者讲自己的事）
  text   句子和 Ellie 问题库的模糊相似度 s∈[0,1]：logit = (s - text_center) / text_scale
         Ellie 说的话几乎都是固定句式，s 高 -> 几乎肯定是 Ellie；s 低不代表是受访者（Ellie 也有库外的话）
  level  句子在原始音频里的响度（有声帧 p90）。两个说话人通常差 7–30 dB，在本 session 的句子响度上拟合两簇
         高斯，哪一簇 Ellie 句子（text 高分）多，哪一簇就是 Ellie；logit = 该簇后验的 logit
  track  句子来自哪一轨（VAD 轨 / 增强后的间隙轨）。间隙轨上的句子和 VAD 轨上的人不同

权重：  logit = w_text * text + w_level(d') * level + w_track * track + bias
  w_level(d') = level_weight * sigmoid((d' - level_d0) / level_tau)
  d' = 两簇响度均值差 / 簇内标准差：差得开（309、676）就信响度，差不开（308）就主要靠文本
"""
from __future__ import annotations

import numpy as np

from ..config import Cfg

HOP = 512


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def segment_levels(wav: np.ndarray, segs: list[dict], sr: int) -> np.ndarray:
    """每句在原始音频里的响度（有声帧 p90）。只看句中每个词的时间区间：
    弱声轨上 WhisperX 的句级时间戳常被拉长（5 个词标成 20 s），整句区间会混进另一个人的声音。"""
    n = len(wav) // HOP
    db = 10 * np.log10((wav[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
    out = []
    for s in segs:
        iv = [(w["start"], w["end"]) for w in s.get("words", []) if w.get("start") is not None
              and w.get("end") is not None and w["end"] - w["start"] < 2.0]
        if not iv:
            iv = [(s["start"], min(s["end"], s["start"] + 2.0))]
        idx = np.concatenate([np.arange(int(a * sr / HOP), max(int(b * sr / HOP), int(a * sr / HOP) + 1))
                              for a, b in iv])
        f = db[idx[idx < n]]
        out.append(float(np.percentile(f, 90)) if len(f) else -100.0)
    return np.asarray(out)


def load_phrases(bank, path) -> list[str]:
    """问题库 + resources/ellie_phrases.txt（只用于说话人判别）。"""
    from pathlib import Path

    from .ellie import normalize

    extra = []
    if path and Path(path).exists():
        extra = [normalize(x) for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith("#")]
    return sorted(set(bank.texts) | set(extra))


def text_scores(segs: list[dict], phrases: list[str]) -> np.ndarray:
    from rapidfuzz import fuzz, process

    from .ellie import normalize, split_sentences

    out = []
    for s in segs:
        best = 0.0
        for sent in split_sentences(s.get("text", "")) or [s.get("text", "")]:
            q = normalize(sent)
            if len(q) < 2:
                continue
            hit = process.extractOne(q, phrases, scorer=fuzz.ratio)
            if hit:
                best = max(best, hit[1] / 100)
        out.append(best)
    return np.asarray(out)


def _gmm2(x: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D 两簇：k-means 硬划分（不会像 EM 那样在两人响度接近时塌成一簇），再用每簇均值 / 标准差算高斯后验。
    返回 (均值, 标准差, 每个点属于第 1 簇的后验)。"""
    if len(x) < 4:
        return np.array([x.mean()] * 2), np.ones(2), np.full(len(x), 0.5)
    mu = np.percentile(x, [25, 75]).astype(float)
    for _ in range(iters):
        lab = np.abs(x[:, None] - mu).argmin(1)
        new = np.array([x[lab == k].mean() if (lab == k).any() else mu[k] for k in (0, 1)])
        if np.allclose(new, mu):
            break
        mu = new
    sd = np.array([x[lab == k].std() if (lab == k).sum() > 1 else 1.0 for k in (0, 1)]) + 1.0
    pi = np.array([(lab == k).mean() for k in (0, 1)]).clip(0.05, 0.95)
    ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(pi)
    r = np.exp(ll - ll.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return mu, sd, r[:, 1]


_FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "my", "me", "myself", "mine"}


def text_features(segs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(是否问句, 第一人称词比例)。Ellie 几乎都在提问；受访者讲自己的事，I / my / me 多。"""
    from .ellie import normalize

    q, fp = [], []
    for s in segs:
        t = s.get("text", "").strip()
        w = normalize(t).split()
        q.append(float(t.endswith("?")))
        fp.append(sum(x in _FIRST_PERSON for x in w) / max(len(w), 1))
    return np.asarray(q), np.asarray(fp)


def score(lvl: np.ndarray, txt: np.ndarray, quiet_track: np.ndarray, sc,
          qmark: np.ndarray | None = None, first_person: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """线索 -> P(Ellie)。sc: speaker 配置（dict 或 Cfg）。"""
    mu, sd, post_hi = _gmm2(lvl)
    hi = int(np.argmax(mu))
    p_hi = post_hi if hi == 1 else 1 - post_hi
    anchor = txt >= sc["anchor_score"]
    # 响度两簇：哪一簇是 Ellie 由 "高分文本句" 投票决定
    ellie_is_loud = bool(p_hi[anchor].mean() > 0.5) if anchor.any() else False
    p_level = p_hi if ellie_is_loud else 1 - p_hi
    dprime = float(abs(mu[1] - mu[0]) / np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2))
    w_level = sc["level_weight"] / (1 + np.exp(-(dprime - sc["level_d0"]) / sc["level_tau"]))
    # 轨道：间隙轨（quiet）上的句子属于 "轻" 的那个人
    track_sign = (2 * quiet_track - 1) * (-1 if ellie_is_loud else 1)
    z = (sc["text_weight"] * (txt - sc["text_center"]) / sc["text_scale"]
         + w_level * _logit(p_level) + sc["track_weight"] * track_sign + sc["bias"])
    if qmark is not None:
        z = z + sc.get("question_weight", 0.0) * (2 * qmark - 1)
    if first_person is not None:
        z = z - sc.get("first_person_weight", 0.0) * first_person / 0.1     # 每 10% 第一人称词

    info = {"level_means_db": mu.tolist(), "dprime": dprime, "w_level": float(w_level),
            "ellie_is_loud": ellie_is_loud, "n_anchor": int(anchor.sum())}
    return 1 / (1 + np.exp(-z)), info


def assign_speakers(segs: list[dict], wav: np.ndarray, bank, cfg: Cfg) -> tuple[list[dict], dict]:
    """给每个 segment 写 speaker = ELLIE / PARTICIPANT（以及 p_ellie 等特征），返回 (segs, info)。"""
    if not segs:
        return segs, {}
    lvl = segment_levels(wav, segs, cfg.preprocessing.sample_rate)
    sc = cfg.preprocessing.speaker
    from ..config import resolve

    txt = text_scores(segs, load_phrases(bank, resolve(sc.phrases)))
    qm, fp = text_features(segs)
    quiet_track = np.array([s.get("track") == "quiet" for s in segs], float)
    p, info = score(lvl, txt, quiet_track, sc, qm, fp)
    for s, pe, lv, t in zip(segs, p, lvl, txt):
        s["speaker"] = "ELLIE" if pe >= 0.5 else "PARTICIPANT"
        s["p_ellie"], s["level_db"], s["text_score"] = float(pe), float(lv), float(t)
        for w in s.get("words", []):
            w["speaker"] = s["speaker"]
    return segs, info


def interview_start(segs: list[dict], bank, min_score: float) -> float:
    """第一句和问题库里 opening 句（"Hi, I'm Ellie" 等）相似度 >= min_score 的 Ellie 句的开始时间；找不到返回 0。"""
    from rapidfuzz import fuzz

    from .ellie import normalize, split_sentences

    opening = [e.text for e in bank.entries if e.type == "opening"]
    for s in segs:
        if s.get("speaker") not in (None, "ELLIE"):
            continue
        for sent in split_sentences(s.get("text", "")) or [s.get("text", "")]:
            q = normalize(sent)
            if q and max((fuzz.partial_ratio(q, o) for o in opening), default=0) / 100 >= min_score \
                    and len(q.split()) >= 3:
                return float(s["start"])
    return 0.0
