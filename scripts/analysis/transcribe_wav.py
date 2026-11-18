"""转写任意一个 wav（如 export_tracks.py 导出的 combined 轨），WhisperX large-v3 + 词级对齐。

文件名是 {sid}_thr{t}_*.wav 时，还会：在原始音频上按阈值 t 重跑 VAD，句子中点落在 VAD 区间里记为 loud 轨、否则 quiet 轨，
再用 speaker.py 判说话人（响度用原始音频），这样和正式流程的线索一致。
输出：同名 .json（segments）和 .txt（可读版：时间 / 说话人 / 文本）。

  sbatch scripts/analysis/transcribe_wav.sbatch data/interim/tracks/309_thr0.7_combined.wav
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing.transcribe import _asr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

cfg = load_config()
sr = cfg.preprocessing.sample_rate
path = Path(a.wav).resolve()
segs = _asr(load_wav(path, sr), "combined", cfg, a.device)

m = re.match(r"(\d+)_thr([\d.]+)_", path.name)
info = {}
if m:
    from src.preprocessing.ellie import QuestionBank
    from src.preprocessing.speaker import assign_speakers
    from src.preprocessing.vad import run_vad

    sid, thr = m.group(1), float(m.group(2))
    raw = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav", sr)
    speech = run_vad(raw, cfg, thr)
    st = np.array([s for s, _ in speech])
    en = np.array([e for _, e in speech])
    for s in segs:
        ws = [((w["start"] + w["end"]) / 2) for w in s.get("words", []) if w.get("start") is not None]
        mids = np.array(ws or [(s["start"] + s["end"]) / 2])
        inside = ((mids[:, None] >= st) & (mids[:, None] <= en)).any(1).mean() if len(st) else 0.0
        s["track"] = "loud" if inside >= 0.5 else "quiet"
    segs, info = assign_speakers(segs, raw, QuestionBank(cfg), cfg)

out = path.with_suffix(".json")
out.write_text(json.dumps({"source": str(path), "speaker": info, "segments": segs}, ensure_ascii=False, indent=1))
lines = [f"[{s['start']:7.1f}-{s['end']:7.1f}] {s.get('speaker', '?'):11s} {s.get('track', ''):5s} "
         f"{s['text'].strip()}" for s in segs]
path.with_suffix(".txt").write_text("\n".join(lines) + "\n")
print(f"{len(segs)} segments -> {out} / {path.with_suffix('.txt')}")
print(json.dumps(info, ensure_ascii=False))
