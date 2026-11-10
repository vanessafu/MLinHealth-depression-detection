"""Step 3: WhisperX transcription + word alignment + 2-speaker diarization."""
from __future__ import annotations

import os

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
        token = os.environ.get(wc.hf_token_env)
        try:  # whisperx >= 3.3.2
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            DiarizationPipeline = whisperx.DiarizationPipeline
        try:  # whisperx >= 3.7（pyannote 4）参数名为 token
            _CACHE["diar"] = DiarizationPipeline(token=token, device=device)
        except TypeError:  # 旧版本参数名为 use_auth_token
            _CACHE["diar"] = DiarizationPipeline(use_auth_token=token, device=device)
    return _CACHE["asr"], _CACHE["align"], _CACHE["diar"]


def transcribe(wav: np.ndarray, cfg: Cfg, device: str) -> dict:
    """Returns whisperx result: {"segments": [{start, end, text, speaker, words: [...]}, ...]}."""
    import whisperx

    wc = cfg.preprocessing.whisperx
    asr, (align_model, meta), diar = _models(cfg, device)
    result = asr.transcribe(wav, batch_size=wc.batch_size, language=wc.language)
    result = whisperx.align(result["segments"], align_model, meta, wav, device,
                            return_char_alignments=False)
    diar_segments = diar(wav, num_speakers=wc.num_speakers)
    result = whisperx.assign_word_speakers(diar_segments, result)
    return {"segments": result["segments"]}


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
