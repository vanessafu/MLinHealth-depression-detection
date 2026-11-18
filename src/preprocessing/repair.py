"""Step 3c（可选）: 用 E-DAIC 官方 XXX_Transcript.csv 修复说话人 + 找回 Ellie 的问题。

官方转写：受访者麦克风上的 ASR，句级时间戳，质量高，几乎全是受访者的话（少数是够响的 Ellie / 工作人员）。
WhisperX 转写（transcribe 步）：两个人都有，但说话人判别和 Ellie 轨的文本不可靠。

做法：
  1. 官方句子里像 Ellie 说法的（问题库 + ellie_phrases 相似度 >= ellie_like_score 且 >= 3 词）-> Ellie；其余 -> 受访者
     受访者区间 P = 这些句子的 [start - pad_s, end + pad_s]
  2. WhisperX 的每句按词切成 "在 P 里 / 不在 P 里" 的连续片段（run，短于 min_run_words 并回相邻段），每个 run：
       support = 和附近（±window_s）官方受访者文本的 n-gram 重合度（>=2 个 bigram 用 bigram 召回率，否则 unigram）
       像 Ellie（>= ellie_like_score）                          -> ELLIE（官方转写里也可能有够响的 Ellie）
       在 P 里 且 support >= support_min                         -> PARTICIPANT（只用来给官方句子提供词级时间）
       不在 P 里 且 support >= support_min                       -> PARTICIPANT（时间戳漂移，同一句话）
       在 P 里 且 support 低：像 Ellie（>= ellie_like_score）-> ELLIE；不像（< weak_like_score）-> PARTICIPANT；
                              介于两者之间 -> UNKNOWN
       不在 P 里 且 support 低（受访者的空隙）                     -> ELLIE，除非：
           像回答（只有 yes/no/um，或第一人称词 >= 12%）且不像 Ellie -> UNKNOWN（官方漏掉的受访者短答）
           对齐分数均值 < halluc_score 且不像 Ellie              -> UNKNOWN（多半是放大噪声后 Whisper 编的）
           响度明显属于受访者那一簇且不像 Ellie                   -> UNKNOWN
  3. 按官方受访者区间再做一条 Ellie 轨（挖掉 P、降噪、逐段放大）重新转写，
     和已有 Ellie 句不重复的补进来（source=gap_asr）
  4. 官方句子里像 Ellie 的、还没被覆盖的，也补进来（source=official_ellie）
  5. Ellie 句子和问题库相似度 >= canon_score 时换成问题库原句（text_asr 保留原始 ASR）
  6. 受访者的话直接用官方句子（source=official），词级时间取 WhisperX 的 PARTICIPANT 词，没有就均匀插值
  7. 访谈开始 = 第一句像开场白的 Ellie 句，之前的丢掉
输出 turns: [{speaker: ELLIE / PARTICIPANT / UNKNOWN, start, end, text, text_asr, source, reason, words}]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Cfg

PART, ELLIE, UNK = "PARTICIPANT", "ELLIE", "UNKNOWN"


# ---------------------------------------------------------------- 官方转写
def official_transcript_path(wav_path: Path, sid: int) -> Path | None:
    for p in (wav_path.parent / f"{sid}_Transcript.csv", wav_path.parent / f"{sid}_P" / f"{sid}_Transcript.csv"):
        if p.exists():
            return p
    return None


def load_official(path: Path) -> list[dict]:
    d = pd.read_csv(path)
    d["Text"] = d["Text"].fillna("").astype(str).str.strip()
    d = d[d["Text"] != ""]
    return [{"start": float(r.Start_Time), "end": float(r.End_Time), "text": r.Text,
             "conf": float(r.Confidence)} for r in d.itertuples()]


# ---------------------------------------------------------------- 小工具
def _tok(text: str) -> list[str]:
    from .ellie import normalize

    return normalize(text).split()


def _grams(t: list[str], n: int) -> set:
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def ngram_support(run_tokens: list[str], ref_tokens: list[str]) -> float:
    """run 里的 n-gram 有多少出现在参考文本里（>=2 个 bigram 用 bigram，否则 unigram）。"""
    if not run_tokens:
        return 0.0
    rb = _grams(run_tokens, 2)
    if len(rb) >= 2:
        return len(rb & _grams(ref_tokens, 2)) / len(rb)
    return sum(w in set(ref_tokens) for w in run_tokens) / len(run_tokens)


def _inside(t: float, iv: np.ndarray) -> bool:
    return bool(len(iv)) and bool(((iv[:, 0] <= t) & (t <= iv[:, 1])).any())


def _overlap_frac(a: float, b: float, ivs: list[tuple[float, float]]) -> float:
    tot = max(b - a, 1e-3)
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in ivs) / tot


def _fill_times(words: list[dict]) -> list[dict]:
    out, last = [], None
    for w in words:
        w = dict(w)
        if w.get("start") is None:
            w["start"] = last if last is not None else 0.0
        if w.get("end") is None:
            w["end"] = w["start"]
        last = w["end"]
        out.append(w)
    return out


_FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "my", "me", "myself", "mine", "we", "we're", "our"}
_ANSWER_WORDS = {"yes", "yeah", "yep", "no", "nope", "nah", "um", "uh", "hmm", "mm", "mhm", "mm-hmm", "uh-huh",
                 "maybe", "sure", "probably", "definitely", "not"}


def _answer_like(toks: list[str]) -> bool:
    """像受访者的回答：全是 yes / no / um 之类，或第一人称词比例 >= 12%。"""
    if not toks:
        return False
    if all(t in _ANSWER_WORDS for t in toks):
        return True
    return sum(t in _FIRST_PERSON for t in toks) / len(toks) >= 0.12


class _EllieLike:
    """问题库 + ellie_phrases 的最高相似度（按句取最大，>= 3 词才算）。"""

    def __init__(self, bank, cfg: Cfg):
        from ..config import resolve
        from .speaker import load_phrases

        self.phrases = load_phrases(bank, resolve(cfg.preprocessing.speaker.phrases))

    def __call__(self, text: str) -> float:
        from rapidfuzz import fuzz, process

        from .ellie import normalize, split_sentences

        best = 0.0
        for s in split_sentences(text) or [text]:
            q = normalize(s)
            if len(q.split()) < 3:
                continue
            hit = process.extractOne(q, self.phrases, scorer=fuzz.ratio)
            best = max(best, hit[1] / 100 if hit else 0.0)
        return best


def canonicalize(text: str, bank, min_score: float) -> tuple[str, int]:
    """Ellie 的句子和问题库原句相似度 >= min_score 就换成原句。返回 (文本, 被替换的句数)。"""
    from rapidfuzz import fuzz, process

    from .ellie import normalize, split_sentences

    out, n = [], 0
    for s in split_sentences(text) or [text]:
        q = normalize(s)
        hit = process.extractOne(q, bank.texts, scorer=fuzz.ratio) if q else None
        if hit and hit[1] >= min_score * 100 and len(q.split()) >= 2:
            e = bank.entries[hit[2]]
            t = e.text[:1].upper() + e.text[1:]
            out.append(t + ("?" if e.type in ("main", "followup") else "."))
            n += 1
        else:
            out.append(s.strip())
    return " ".join(out), n


# ---------------------------------------------------------------- 主流程
def repair_session(wav: np.ndarray, before: dict, official: list[dict], cfg: Cfg, device: str) -> dict:
    from .ellie import QuestionBank
    from .speaker import interview_start, segment_levels

    rc = cfg.preprocessing.repair
    sr = cfg.preprocessing.sample_rate
    bank = QuestionBank(cfg)
    ellie_like = _EllieLike(bank, cfg)

    # 1. 官方句子：Ellie / 受访者
    for o in official:
        o["ellie_like"] = ellie_like(o["text"])
        o["role"] = ELLIE if o["ellie_like"] >= rc.ellie_like_score else PART
    part = [o for o in official if o["role"] == PART]
    p_iv = np.array([(o["start"] - rc.pad_s, o["end"] + rc.pad_s) for o in part]).reshape(-1, 2)
    part_tok = [(o["start"], o["end"], _tok(o["text"])) for o in part]

    def local_ref(a: float, b: float) -> list[str]:
        return [t for s, e, toks in part_tok if e >= a - rc.window_s and s <= b + rc.window_s for t in toks]

    # 响度：before 里两簇的均值 + 哪一簇是 Ellie
    sp = before.get("speaker", {})
    mu = sp.get("level_means_db")
    mu_p = mu_e = None
    if mu and sp.get("dprime", 0) >= rc.level_min_dprime:
        lo, hi = min(mu), max(mu)
        mu_e, mu_p = (hi, lo) if sp.get("ellie_is_loud") else (lo, hi)

    # 2. WhisperX 句子按词切成 run，逐个判
    runs = []
    for seg in before["segments"]:
        words = _fill_times(seg.get("words", []))
        if not words:
            continue
        # 按 "词中点在不在受访者区间里" 切段；短于 min_run_words 的段并回相邻段（区间边界会把一句话切碎）
        flags = [_inside((w["start"] + w["end"]) / 2, p_iv) for w in words]
        parts: list[list[int]] = []
        for i, f in enumerate(flags):
            if parts and flags[parts[-1][0]] == f:
                parts[-1].append(i)
            else:
                parts.append([i])
        changed = True
        while changed and len(parts) > 1:
            changed = False
            for k, pt in enumerate(parts):
                if len(pt) < rc.min_run_words:
                    j = k - 1 if k > 0 else k + 1
                    lo_, hi_ = sorted((j, k))
                    parts[lo_:hi_ + 1] = [parts[lo_] + parts[hi_]]
                    changed = True
                    break
        for pt in parts:
            ws = [words[i] for i in pt]
            runs.append({"words": ws, "in_p": bool(np.mean([flags[i] for i in pt]) >= 0.5),
                         "track": seg.get("track"), "before_speaker": seg.get("speaker")})
    for r in runs:
        ws = r["words"]
        r["start"], r["end"] = ws[0]["start"], ws[-1]["end"]
        r["text"] = " ".join(w["word"].strip() for w in ws).strip()
        toks = _tok(r["text"])
        r["support"] = ngram_support(toks, local_ref(r["start"], r["end"]))
        r["ellie_like"] = ellie_like(r["text"])
        sc = [w["score"] for w in ws if w.get("score") is not None]
        r["align"] = float(np.mean(sc)) if sc else 1.0
        like = r["ellie_like"] >= rc.ellie_like_score
        weak_like = r["ellie_like"] >= rc.weak_like_score
        if like:                                        # 官方转写里也会混进够响的 Ellie（308 整段开场白）
            r["speaker"], r["reason"] = ELLIE, "ellie_like"
        elif r["support"] >= rc.support_min:
            r["speaker"], r["reason"] = PART, "official_ngram"
        elif r["in_p"]:
            if not weak_like:                           # 不像 Ellie：同一段受访者语音的另一种 ASR 结果
                r["speaker"], r["reason"] = PART, "inside_p_not_ellie_like"
            else:
                r["speaker"], r["reason"] = UNK, "inside_p_ambiguous"
        else:
            r["speaker"], r["reason"] = ELLIE, "gap"
            if not weak_like and _answer_like(toks):
                r["speaker"], r["reason"] = UNK, "gap_answer_like"   # 官方转写漏掉的受访者短答 / 自述
            elif r["align"] < rc.halluc_score and not weak_like:
                r["speaker"], r["reason"] = UNK, "gap_low_align"
            elif mu_p is not None and not weak_like:
                lv = float(segment_levels(wav, [r], sr)[0])
                if abs(lv - mu_p) + rc.level_margin_db < abs(lv - mu_e):
                    r["speaker"], r["reason"] = UNK, "gap_participant_loudness"

    turns = []
    for r in runs:
        if r["speaker"] == PART:
            continue                                     # 受访者的话用官方句子，WhisperX 的只提供词时间
        src = "whisperx" if r["before_speaker"] == ELLIE else "relabeled"
        turns.append({"speaker": r["speaker"], "start": r["start"], "end": r["end"], "text": r["text"],
                      "text_asr": r["text"], "source": src if r["speaker"] == ELLIE else "whisperx",
                      "reason": r["reason"], "words": r["words"],
                      "support": round(r["support"], 3), "ellie_like": round(r["ellie_like"], 3),
                      "align": round(r["align"], 3)})

    # 3. 按官方受访者区间重做 Ellie 轨并转写
    n_gap = 0
    if rc.gap_asr:
        from .enhance import ellie_track
        from .transcribe import _asr

        et = dict(cfg.preprocessing.ellie_track)
        et["pad_s"] = 0.0                                 # p_iv 已经带了 pad
        g_wav, _ = ellie_track(wav, sr, [tuple(x) for x in p_iv], **et)
        from rapidfuzz import fuzz

        from .ellie import normalize

        for seg in _asr(g_wav, "gap", cfg, device):
            words = _fill_times(seg.get("words", []))
            if not words:
                continue
            a, b = words[0]["start"], words[-1]["end"]
            text = seg.get("text", "").strip()
            q = normalize(text)
            if not q:
                continue
            ell = [(t["start"], t["end"]) for t in turns if t["speaker"] == ELLIE]
            if _overlap_frac(a, b, ell) >= 0.3 or any(
                    abs(t["start"] - a) <= 10 and fuzz.ratio(q, normalize(t["text_asr"])) >= 80
                    for t in turns if t["speaker"] == ELLIE):
                continue
            if ngram_support(q.split(), local_ref(a, b)) >= rc.support_min:
                continue                                  # 受访者的尾音漏进来了
            sc = [w["score"] for w in words if w.get("score") is not None]
            al = float(np.mean(sc)) if sc else 1.0
            like = ellie_like(text)
            ok = al >= rc.halluc_score or like >= rc.weak_like_score
            turns.append({"speaker": ELLIE if ok else UNK, "start": a, "end": b, "text": text, "text_asr": text,
                          "source": "gap_asr", "reason": "gap_asr" if ok else "gap_asr_low_align",
                          "words": words, "support": 0.0, "ellie_like": round(like, 3), "align": round(al, 3)})
            n_gap += ok

    # 4. 官方句子里的 Ellie
    for o in official:
        if o["role"] != ELLIE:
            continue
        ell = [(t["start"], t["end"]) for t in turns if t["speaker"] == ELLIE]
        if _overlap_frac(o["start"], o["end"], ell) >= 0.3:
            continue
        turns.append({"speaker": ELLIE, "start": o["start"], "end": o["end"], "text": o["text"],
                      "text_asr": o["text"], "source": "official_ellie", "reason": "official_ellie_like",
                      "words": _even_words(o), "support": 0.0, "ellie_like": round(o["ellie_like"], 3),
                      "align": None})

    # 5. Ellie 句换成问题库原句
    for t in turns:
        if t["speaker"] == ELLIE:
            t["text"], t["n_canonical"] = canonicalize(t["text_asr"], bank, rc.canon_score)

    # 6. 受访者 = 官方句子 + WhisperX 词时间
    p_words = [w for r in runs if r["speaker"] == PART for w in r["words"]]
    for o in part:
        ws = [w for w in p_words if o["start"] - rc.pad_s <= (w["start"] + w["end"]) / 2 <= o["end"] + rc.pad_s]
        turns.append({"speaker": PART, "start": o["start"], "end": o["end"], "text": o["text"],
                      "text_asr": " ".join(w["word"].strip() for w in ws), "source": "official",
                      "reason": "official", "words": ws or _even_words(o), "conf": o["conf"]})

    turns.sort(key=lambda t: (t["start"], t["end"]))
    # 7. 访谈开始
    t0 = interview_start(turns, bank, cfg.preprocessing.speaker.opening_min_score)
    turns = [t for t in turns if t["end"] > t0 - 0.5]
    from collections import Counter

    stats = {"n_turns": len(turns), "by_speaker": dict(Counter(t["speaker"] for t in turns)),
             "by_source": dict(Counter(f"{t['speaker']}:{t['source']}" for t in turns)),
             "gap_asr_added": n_gap, "official_ellie_like": sum(o["role"] == ELLIE for o in official),
             "n_official": len(official)}
    return {"interview_start": t0, "turns": turns, "stats": stats}


def _even_words(o: dict) -> list[dict]:
    toks = o["text"].split()
    if not toks:
        return []
    step = (o["end"] - o["start"]) / len(toks)
    return [{"word": w, "start": o["start"] + i * step, "end": o["start"] + (i + 1) * step, "interp": True}
            for i, w in enumerate(toks)]


def turns_to_utterances(rep: dict) -> list[dict]:
    """给 qa 步用：去掉 UNKNOWN。"""
    return [{"speaker": t["speaker"], "start": t["start"], "end": t["end"], "text": t["text"], "words": t["words"],
             "source": t["source"]} for t in rep["turns"] if t["speaker"] != UNK]
