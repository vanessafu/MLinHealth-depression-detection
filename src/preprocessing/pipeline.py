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
        from .snr import analyze
        from .vad import run_vad

        vc = cfg.preprocessing.vad
        _dump({"duration": len(get_wav()) / sr, "snr": analyze(get_wav(), cfg),
               "speech_by_threshold": {str(t): run_vad(get_wav(), cfg, t) for t in vc.thresholds}}, p_vad)

    # --- WhisperX ---
    p_tr = interim_path(cfg, "transcripts", sid, "json")
    method = cfg.preprocessing.transcribe.method
    if "transcribe" in steps and p_tr.exists() and not overwrite:
        made = _load(p_tr).get("method", "two_track")     # two_track 的转写没写 method 字段
        if made != method:
            raise ValueError(f"{p_tr} 是 {made} 的转写，配置是 {method}：换方法请用对应配置"
                             f"（config/two_track.yaml 有单独的 interim 目录）或加 --overwrite")
    if "transcribe" in steps and (overwrite or not p_tr.exists()):
        if method == "compressed":
            from .transcribe import transcribe_compressed

            _dump(transcribe_compressed(get_wav(), cfg, device), p_tr)
        elif method == "two_track":
            from .transcribe import transcribe

            _dump(transcribe(get_wav(), _load(p_vad)["speech_by_threshold"], cfg, device), p_tr)
        else:
            raise ValueError(f"unknown preprocessing.transcribe.method: {method}")

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
