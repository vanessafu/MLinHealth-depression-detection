"""Step 3: 转写 + 逐句说话人判别。preprocessing.transcribe.method 选方法：
  compressed（默认）：transcribe_compressed，见该函数说明
  two_track：transcribe，下面是它的做法

two_track：分轨转写（WhisperX ASR + 词级对齐，不用 pyannote）+ 多阈值融合 + 逐句说话人判别。

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


def _faster_whisper(cfg: Cfg, device: str):
    if "fw" not in _CACHE:
        from faster_whisper import WhisperModel

        wc = cfg.preprocessing.whisperx
        _CACHE["fw"] = WhisperModel(wc.model, device=device, compute_type=wc.compute_type)
    return _CACHE["fw"]


def transcribe_compressed(wav: np.ndarray, cfg: Cfg, device: str) -> dict:
    """method = compressed：不分轨、不降噪，两路转写词级合并 + 声纹判说话人（src/preprocessing/{compress,
    asr_merge,voiceprint}.py；5 个 session 上的对比：官方转写召回 0.967 vs two_track 0.958）。

      comp : 原始 wav 逐帧压缩（轻的人抬上来）-> faster-whisper 无 VAD -> WhisperX 对齐       主转写
      raw  : 原始 wav -> WhisperX（pyannote VAD，chunk_size 秒的块）-> 对齐                   补漏 + 漂移参考
      novad: 原始 wav -> faster-whisper 无 VAD -> 对齐                                       只当漂移参考
    合并：comp 为主；comp 里时间戳漂移的段降级到 raw 之后再补；去幻觉、折叠循环。
    """
    import whisperx

    from .asr_merge import merge_passes, to_units
    from .compress import compress
    from .ellie import QuestionBank
    from .speaker import interview_start, load_phrases, text_scores
    from .voiceprint import assign_speakers_voiceprint
    from ..config import resolve

    wc, tc = cfg.preprocessing.whisperx, cfg.preprocessing.transcribe
    sr = cfg.preprocessing.sample_rate
    asr, (align_model, meta) = _models(cfg, device)
    fw = _faster_whisper(cfg, device)

    def align(segs, audio):
        return whisperx.align(segs, align_model, meta, audio, device, return_char_alignments=False)["segments"]

    def no_vad(audio):
        it, _ = fw.transcribe(audio, language=wc.language, vad_filter=False, beam_size=5,
                              condition_on_previous_text=False)
        return align([{"start": s.start, "end": s.end, "text": s.text} for s in it], audio)

    comp_wav, comp_info = compress(wav, sr, **tc.compress)
    passes = {
        "comp": no_vad(comp_wav),
        "raw": align(asr.transcribe(wav, batch_size=wc.batch_size, language=wc.language,
                                    chunk_size=tc.chunk_size)["segments"], wav),
        "novad": no_vad(wav),
    }
    units, merge_info = merge_passes(passes, ["comp", "raw"], refs=["raw", "novad"],
                                     min_word_score=tc.min_word_score, dup_win=tc.dup_win)
    bank = QuestionBank(cfg)
    phrases = load_phrases(bank, resolve(cfg.preprocessing.speaker.phrases))
    for u, s in zip(units, text_scores(units, phrases)):     # 和 Ellie 台词库的相似度，判说话人用
        u["text_score"] = float(s)

    # 访谈开始：在各路原始转写里找开场白取最早（不受合并 / 漂移保护影响）
    all_u = sorted((u for k in ("comp", "raw", "novad") for u in to_units(passes[k], k, 2.0)), key=lambda u: u["start"])
    t0 = interview_start(all_u, bank, cfg.preprocessing.speaker.opening_min_score)
    units = [u for u in units if u["end"] > t0 - 0.5]

    emb_audio = comp_wav if tc.speaker.embed_audio == "compressed" else wav
    spk_info = assign_speakers_voiceprint(units, wav, emb_audio, cfg, device)
    return {"segments": units, "interview_start": t0, "method": "compressed", "compress": comp_info,
            "merge": merge_info, "speaker": spk_info,
            "n_pass_segments": {k: len(v) for k, v in passes.items()}}


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
