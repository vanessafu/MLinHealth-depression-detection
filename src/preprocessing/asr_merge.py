"""多路 ASR 结果的词级合并：切单元、去幻觉、折叠循环、漂移保护、按优先级补漏。

单元（unit）= {"start", "end", "text", "words": [{"word", "start", "end", "score"}], "track", "word_score"}，
时间都用词级时间（WhisperX 句级时间常被拉长）。
"""
from __future__ import annotations

import re

import numpy as np

from .ellie import normalize

# Whisper 在静音 / 噪声上的经典幻觉句
HALLUCINATIONS = ["thanks for watching", "thank you for watching", "please subscribe", "subscribe to",
                  "see you in the next video", "subtitles by", "transcribed by", "amara org"]


def is_hallucination(text: str) -> bool:
    t = normalize(text)
    return any(h in t for h in HALLUCINATIONS)


def to_units(segs: list[dict], track: str, gap: float) -> list[dict]:
    """WhisperX segments -> 单元：按句末标点和 > gap 秒的词间停顿切开。没有时间的词（数字等）继承前一个词。"""
    groups = []
    for s in segs:
        ws = [dict(w) for w in s.get("words", []) if w.get("word", "").strip()]
        for i, w in enumerate(ws):
            if w.get("start") is None or w.get("end") is None:
                prev = next((ws[j]["end"] for j in range(i - 1, -1, -1) if ws[j].get("end") is not None), s["start"])
                w["start"], w["end"] = prev, prev
                w.setdefault("score", 0.5)
        cur: list[dict] = []
        for w in ws:
            if cur and (w["start"] - cur[-1]["end"] > gap or re.search(r"[.?!]$", cur[-1]["word"])):
                groups.append(cur)
                cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
    out = []
    for ws in groups:
        words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                  "score": float(w.get("score", 0.5))} for w in ws]
        out.append({"start": words[0]["start"], "end": max(w["end"] for w in words),
                    "text": " ".join(w["word"].strip() for w in words), "words": words, "track": track,
                    "word_score": float(np.mean([w["score"] for w in words]))})
    return out


def word_iv(u: dict) -> list[tuple[float, float]]:
    return [(w["start"], w["end"]) for w in u["words"]]


def _covered(a: float, b: float, iv: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, d) - max(a, c)) for c, d in iv) / max(b - a, 0.02)


def add_source(accepted: list[dict], segs: list[dict], track: str, *, gap: float, min_word_score: float,
               dup_win: float, info: dict) -> list[dict]:
    """词级合并：segs 里已被 accepted 覆盖（时间重叠 >= 50%）的词丢掉，剩下的词重新切单元；
    再丢对齐分数过低的单元、和 dup_win 秒内已有相似文本的单元（同一句话两路转写时间可能差好几秒）。"""
    from rapidfuzz import fuzz

    iv = [x for u in accepted for x in word_iv(u)]
    rest, n_cov = [], 0
    for s in segs:
        ws = [w for w in s.get("words", []) if w.get("start") is not None and w.get("end") is not None]
        keep = [w for w in ws if _covered(w["start"], w["end"], iv) < 0.5]
        n_cov += len(ws) - len(keep)
        if keep:
            rest.append({**s, "words": keep})
    new, n_low, n_dup = [], 0, 0
    for u in to_units(rest, track, gap):
        t = normalize(u["text"])
        if u["word_score"] < min_word_score:
            n_low += 1
            continue
        near = " ".join(normalize(v["text"]) for v in accepted
                        if v["end"] > u["start"] - dup_win and v["start"] < u["end"] + dup_win)
        if len(t) > 8 and near and fuzz.partial_ratio(t, near) >= 85:
            n_dup += 1
            continue
        new.append(u)
    info[track] = {"covered_words": n_cov, "low_score": n_low, "dup_text": n_dup, "added": len(new)}
    return sorted(accepted + new, key=lambda u: u["start"])


def collapse_repeats(segs: list[dict], ref_segs: list[dict], min_run: int = 3, win: float = 20.0) -> tuple[list, int]:
    """Whisper 循环：同一句（>= 2 个词）连续 >= min_run 次 -> 只留第一句；
    参考转写（ref_segs）同一时间窗里也有这种重复就不动（"I'm trying" x7 是受访者真这么说的）。"""
    from rapidfuzz import fuzz

    out, i, n_drop = [], 0, 0
    while i < len(segs):
        t = normalize(segs[i].get("text", ""))
        j = i + 1
        while j < len(segs) and normalize(segs[j].get("text", "")) == t and segs[j]["start"] - segs[i]["start"] < win:
            j += 1
        if j - i >= min_run and len(t.split()) >= 2:
            ref = " ".join(normalize(s.get("text", "")) for s in ref_segs
                           if s["end"] > segs[i]["start"] - 5 and s["start"] < segs[j - 1]["end"] + 5)
            if not (ref.count(t) >= 2 or fuzz.partial_ratio(f"{t} {t}", ref) >= 90):
                out.append(segs[i])
                n_drop += j - i - 1
                i = j
                continue
        out.extend(segs[i:j])
        i = j
    return out, n_drop


def drift_guard(segs: list[dict], refs: list[list[dict]], *, bad_s: float = 2.0, good_s: float = 1.5,
                win: float = 15.0, min_bad: int = 2, cluster_s: float = 20.0, pre_s: float = 3.0,
                post_s: float = 10.0) -> tuple[list[dict], list[dict]]:
    """无 VAD 长窗解码偶尔整段时间戳漂移（308 开场白比原始 WhisperX 早 5–12 s）。
    和任一参考转写文本对得上的句子：起点差 <= good_s 为 good，所有对得上的都 > bad_s 为 bad，其余 none。
    bad 句相邻 <= cluster_s 成簇、簇里 >= min_bad 句才算漂移（Ellie 会重复同一句话，单句 bad 常是匹配到了另一次）；
    [簇首 - pre_s, 簇尾 + post_s] 内非 good 的句子降级。返回 (保留, 降级)。
    降级而不是删：漂移段里可能有只有这一路转出来的内容（300 的 104–128 s），排到参考转写之后再补。"""
    from rapidfuzz import fuzz

    rns = [[(s["start"], normalize(s.get("text", ""))) for s in r] for r in refs]
    status = []
    for s in segs:
        t = normalize(s.get("text", ""))
        st = "none"
        if len(t) >= 10:
            offs = [abs(rs - s["start"]) for rn in rns for rs, rt in rn
                    if abs(rs - s["start"]) <= win and fuzz.partial_ratio(t, rt) >= 85]
            if offs:
                st = "good" if min(offs) <= good_s else ("bad" if min(offs) > bad_s else "none")
        status.append(st)
    clusters, cur = [], []
    for i in (i for i, st in enumerate(status) if st == "bad"):
        if cur and segs[i]["start"] - segs[cur[-1]]["start"] > cluster_s:
            clusters.append(cur)
            cur = []
        cur.append(i)
    if cur:
        clusters.append(cur)
    drop = np.zeros(len(segs), bool)
    for c in clusters:
        if len(c) < min_bad:
            continue
        lo, hi = segs[c[0]]["start"] - pre_s, segs[c[-1]]["end"] + post_s
        for j, s in enumerate(segs):
            if lo <= s["start"] <= hi and status[j] != "good":
                drop[j] = True
    return [s for s, d in zip(segs, drop) if not d], [s for s, d in zip(segs, drop) if d]


def merge_passes(passes: dict[str, list[dict]], order: list[str], refs: list[str], *, gap: float = 0.8,
                 min_word_score: float = 0.3, dup_win: float = 15.0, **drift_kw) -> tuple[list[dict], dict]:
    """order[0] 为主，其余按顺序补漏。refs 里的路不做漂移保护（它们是参考）；
    其他路的漂移段降级到 refs 里最后一路之后再补。"""
    info: dict = {"hallucination": 0, "repeat_collapsed": {}, "drift_demoted": {}}
    clean, demoted = {}, {}
    for k in order:
        segs = [s for s in passes[k] if not is_hallucination(s.get("text", ""))]
        info["hallucination"] += len(passes[k]) - len(segs)
        segs, n = collapse_repeats(segs, passes[refs[0]])
        info["repeat_collapsed"][k] = n
        if k not in refs:
            segs, demoted[k] = drift_guard(segs, [passes[r] for r in refs], **drift_kw)
            info["drift_demoted"][k] = len(demoted[k])
        clean[k] = segs
    seq = [(k, clean[k]) for k in order]
    last_ref = max(i for i, (k, _) in enumerate(seq) if k in refs)
    seq[last_ref + 1:last_ref + 1] = [(f"{k}_drift", v) for k, v in demoted.items() if v]
    units = to_units(seq[0][1], seq[0][0], 2.0)
    for k, segs in seq[1:]:
        units = add_source(units, segs, k, gap=gap, min_word_score=min_word_score, dup_win=dup_win, info=info)
    return units, info

