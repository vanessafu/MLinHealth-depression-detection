"""Run preprocessing for one session with per-step caching.

Steps: vad -> transcribe -> qa (Ellie id + classify + Q-A pairs + trim) -> segments -> wavlm
Each step reads the previous step's file and is skipped if its output exists (unless overwrite).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..audio import load_wav
from ..config import Cfg
from . import feature_path, interim_path

STEPS = ["vad", "transcribe", "qa", "segments", "wavlm"]


def _dump(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def process_session(sid: int, wav_path: Path, cfg: Cfg, steps: list[str], device: str,
                    overwrite: bool = False) -> dict:
    sr = cfg.preprocessing.sample_rate
    status = {"sid": sid}
    wav = None

    def get_wav():
        nonlocal wav
        if wav is None:
            wav = load_wav(wav_path, sr)
        return wav

    # --- VAD ---
    p_vad = interim_path(cfg, "vad", sid, "json")
    if "vad" in steps and (overwrite or not p_vad.exists()):
        from .vad import run_vad

        _dump({"duration": len(get_wav()) / sr, "speech": run_vad(get_wav(), cfg)}, p_vad)

    # --- WhisperX ---
    p_tr = interim_path(cfg, "transcripts", sid, "json")
    if "transcribe" in steps and (overwrite or not p_tr.exists()):
        from .transcribe import transcribe
        from .vad import mask_non_speech

        vad = _load(p_vad)
        masked = mask_non_speech(get_wav(), vad["speech"], sr)
        _dump(transcribe(masked, cfg, device), p_tr)

    # --- Ellie + Q-A pairs ---
    p_qa = interim_path(cfg, "qa_pairs", sid, "json")
    if "qa" in steps and (overwrite or not p_qa.exists()):
        from .ellie import QuestionBank, classify_ellie_sentences, identify_ellie
        from .qa_pairs import build_qa_pairs
        from .transcribe import words_to_utterances

        bank = QuestionBank(cfg)
        utts = words_to_utterances(_load(p_tr), cfg.preprocessing.whisperx.utterance_gap_sec)
        ellie, counts = identify_ellie(utts, bank)
        if ellie is None:
            status["error"] = "Ellie not identified (no bank matches)"
            return status
        events = classify_ellie_sentences(utts, ellie, bank, cfg)
        pairs, ellie_iv = build_qa_pairs(events, cfg)
        _dump({"sid": sid, "ellie_speaker": ellie, "speaker_match_counts": counts,
               "duration": _load(p_vad)["duration"], "ellie_intervals": ellie_iv,
               # 问题库里没有的 Ellie 句子，用来补全 resources/ellie_questions.csv
               "unmatched_ellie": [e["text"] for e in events if e["role"] == "ellie" and not e["matched"]],
               "pairs": pairs}, p_qa)

    # --- segments ---
    p_seg = interim_path(cfg, "segments", sid, "csv")
    if "segments" in steps and (overwrite or not p_seg.exists()):
        from .segments import make_segments

        qa = _load(p_qa)
        segs = make_segments(qa["pairs"], [tuple(x) for x in qa["ellie_intervals"]], qa["duration"], cfg)
        pd.DataFrame(segs, columns=["seg_id", "pair_id", "start", "end", "pos"]).to_csv(p_seg, index=False)

    # --- WavLM cache ---
    p_feat = feature_path(cfg, sid)
    if "wavlm" in steps and (overwrite or not p_feat.exists()):
        import torch

        from .wavlm_features import extract

        segs = pd.read_csv(p_seg).to_dict("records")
        torch.save(extract(get_wav(), segs, cfg, device), p_feat)

    status["ok"] = True
    return status


def load_qa_pairs(cfg: Cfg, sid: int) -> list[dict]:
    return _load(interim_path(cfg, "qa_pairs", sid, "json"))["pairs"]
