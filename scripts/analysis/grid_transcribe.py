"""网格：每个 session × 每个 VAD 阈值跑一次分轨转写（不判说话人），缓存到 data/interim/grid/，供 tune_speaker.py 调参。
auto = snr.analyze 的 silero_otsu（试过，不如两阈值融合）。

  sbatch ... python scripts/analysis/grid_transcribe.py --sids 308 309 676 --thresholds 0.12 0.45 0.7
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing.enhance import ellie_track  # noqa: E402
from src.preprocessing.snr import analyze  # noqa: E402
from src.preprocessing.transcribe import _asr  # noqa: E402
from src.preprocessing.vad import mask_non_speech, run_vad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sids", nargs="*", type=int, default=[308, 309, 676])
ap.add_argument("--thresholds", nargs="*", default=["auto", "0.12", "0.3", "0.45", "0.6", "0.7"])
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
cfg = load_config()
out = ROOT / "data/interim/grid"
out.mkdir(parents=True, exist_ok=True)
for sid in a.sids:
    wav = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav")
    snr = analyze(wav, cfg)
    for t in a.thresholds:
        p = out / f"{sid}_thr{t}.json"
        if p.exists():
            continue
        thr = float(np.clip(snr["silero_otsu"], 0.3, 0.7)) if t == "auto" else float(t)
        t0 = time.time()
        speech = run_vad(wav, cfg, thr)
        loud = _asr(mask_non_speech(wav, speech, 16000), "loud", cfg, a.device)
        q_wav, _ = ellie_track(wav, 16000, speech, **cfg.preprocessing.ellie_track)
        segs = sorted(loud + _asr(q_wav, "quiet", cfg, a.device), key=lambda s: s["start"])
        tr = {"segments": segs, "threshold": thr, "snr": snr, "speech": speech}
        json.dump(tr, open(p, "w"), ensure_ascii=False, indent=1)
        n = {k: sum(s["track"] == k for s in segs) for k in ("loud", "quiet")}
        print(f"{sid} thr={t}({thr:.2f}) {n} {time.time() - t0:.0f}s", flush=True)
