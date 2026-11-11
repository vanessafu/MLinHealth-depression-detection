"""Step 3: 分轨转写（WhisperX ASR + 词级对齐，不用 pyannote）+ 多阈值融合 + 逐句说话人判别。

  对每个 VAD 阈值：loud 轨 = 原始音频只留 VAD 区间；quiet 轨 = VAD 区间挖掉后降噪 + 逐段 RMS 标准化（enhance.py）
  多个阈值的结果融合：以第一个（低）阈值为主，后面的（高）阈值只补前面没有的内容（fuse_runs）
    低阈值：响的人 + 一部分轻的人都在 loud 轨里，不经增强，Whisper 听得最完整（308 / 309 最好）
    高阈值：轻的人更多落到 quiet 轨被增强（676 这种整体很小声的录音最好）
  说话人：speaker.assign_speakers 按 文本 / 响度 / 轨道 / 问句 加权逐句判别（录音里谁响谁轻因 session 而异）
  访谈开始 = 第一句像 Ellie 开场白的句子，之前的（工作人员调试设备）丢掉
"""
from __future__ import annotations

import numpy as np

from ..config import Cfg

_CACHE: dict = {}


def _models(cfg: Cfg, device: str):
    import whisperx

    wc = cfg.preprocessing.whisperx
    if "asr" not in _CACHE:
        _CACHE["asr"] = whisperx.load_model(wc.model, device, compute_type=wc.compute_type,
                                            language=wc.language)
        _CACHE["align"] = whisperx.load_align_model(language_code=wc.language, device=device)
    return _CACHE["asr"], _CACHE["align"]


def _asr(wav: np.ndarray, track: str, cfg: Cfg, device: str) -> list[dict]:
    import whisperx

    wc = cfg.preprocessing.whisperx
    asr, (align_model, meta) = _models(cfg, device)
    r = asr.transcribe(wav, batch_size=wc.batch_size, language=wc.language)
    r = whisperx.align(r["segments"], align_model, meta, wav, device, return_char_alignments=False)
    segs = r["segments"]
    for seg in segs:
        seg["track"] = track
    return segs


def _word_iv(seg: dict) -> list[tuple[float, float]]:
    return [(w["start"], w["end"]) for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None]


def fuse_runs(base: list[dict], extra: list[dict], max_overlap: float = 0.3, text_win_s: float = 30.0,
              text_sim: float = 80.0) -> tuple[list[dict], int]:
    """以 base（低 VAD 阈值）为主，把 extra（高阈值）里 base 没有的内容补进来。
    重复判定：词级时间被 base 覆盖 >= max_overlap，或 text_win_s 秒内 base 有文本相似度 >= text_sim 的句子
    （弱声段上同一句话在两次运行里的时间戳可能差十几秒，所以还要看文本）。"""
    from rapidfuzz import fuzz

    from .ellie import normalize

    iv = [x for s in base for x in _word_iv(s)]
    bt = [(s["start"], normalize(s.get("text", ""))) for s in base]
    added = []
    for s in extra:
        ws = _word_iv(s)
        tot = sum(max(b - a, 0.02) for a, b in ws)
        cov = sum(max(0.0, min(b, d) - max(a, c)) for a, b in ws for c, d in iv) / tot if ws else 1.0
        if cov >= max_overlap:
            continue
        q = normalize(s.get("text", ""))
        if not q or any(abs(st - s["start"]) <= text_win_s and t and fuzz.ratio(q, t) >= text_sim for st, t in bt):
            continue
        added.append(s)
    return sorted(base + added, key=lambda x: x["start"]), len(added)


def transcribe(wav: np.ndarray, speech_by_threshold: dict[str, list], cfg: Cfg, device: str) -> dict:
    """Returns {"segments": [{start, end, text, speaker, track, p_ellie, words}, ...], "interview_start", ...}."""
    from .ellie import QuestionBank
    from .enhance import ellie_track
    from .speaker import assign_speakers, interview_start
    from .vad import mask_non_speech

    sr = cfg.preprocessing.sample_rate
    runs, info = [], {}
    for thr in cfg.preprocessing.vad.thresholds:
        speech = speech_by_threshold[str(thr)]
        loud = _asr(mask_non_speech(wav, speech, sr), "loud", cfg, device)
        q_wav, q_info = ellie_track(wav, sr, speech, **cfg.preprocessing.ellie_track)
        runs.append(sorted(loud + _asr(q_wav, "quiet", cfg, device), key=lambda s: s["start"]))
        info[f"quiet_track@{thr}"] = q_info
    segs = runs[0]
    for extra in runs[1:]:
        segs, n = fuse_runs(segs, extra)
        info.setdefault("fused_added", []).append(n)
    bank = QuestionBank(cfg)
    segs, info["speaker"] = assign_speakers(segs, wav, bank, cfg)
    t0 = interview_start(segs, bank, cfg.preprocessing.speaker.opening_min_score)
    segs = [s for s in segs if s["end"] > t0 - 0.5]
    return {"segments": segs, "interview_start": t0, **info}


def words_to_utterances(result: dict, gap_sec: float) -> list[dict]:
    """Flatten words and regroup into single-speaker utterances.

    WhisperX segments can contain several speakers; regroup by word-level speaker labels.
    Words without timestamps (numbers etc.) inherit the neighbour's times.
    """
    words = []
    for seg in result["segments"]:
        seg_spk = seg.get("speaker")
        for w in seg.get("words", []):
            words.append({"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end"),
                          "speaker": w.get("speaker", seg_spk)})
    # fill missing timestamps from neighbours
    for i, w in enumerate(words):
        if w["start"] is None:
            prev_end = words[i - 1]["end"] if i > 0 and words[i - 1]["end"] is not None else None
            w["start"] = prev_end if prev_end is not None else 0.0
        if w["end"] is None:
            w["end"] = w["start"]
    utts: list[dict] = []
    for w in words:
        cur = utts[-1] if utts else None
        if cur and cur["speaker"] == w["speaker"] and w["start"] - cur["end"] <= gap_sec:
            cur["words"].append(w)
            cur["end"] = w["end"]
        else:
            utts.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w]})
    for u in utts:
        u["text"] = " ".join(x["word"].strip() for x in u["words"]).strip()
    return utts
