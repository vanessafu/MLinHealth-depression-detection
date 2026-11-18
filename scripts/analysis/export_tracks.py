"""导出转写前的两条轨（按 config 里当前的 VAD / ellie_track 参数），用来试听。

每个 session、每个 VAD 阈值写 3 个 wav 到 data/interim/tracks/：
  {sid}_thr{t}_loud.wav      原始音频只保留 VAD 区间（= loud 轨，Whisper 实际听到的）
  {sid}_thr{t}_quiet.wav     Ellie 轨：VAD 区间(+pad) 挖掉，其余降噪 + 逐段 RMS 标准化（= quiet 轨）
  {sid}_thr{t}_combined.wav  两者相加（两条轨在时间上不重叠），时间轴和原始音频一致

  python scripts/analysis/export_tracks.py --sids 308
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing.enhance import ellie_track  # noqa: E402
from src.preprocessing.vad import mask_non_speech, run_vad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sids", nargs="*", type=int, default=[308])
ap.add_argument("--thresholds", nargs="*", type=float, default=None, help="默认用 config 里的 vad.thresholds")
a = ap.parse_args()

cfg = load_config()
sr = cfg.preprocessing.sample_rate
vc = cfg.preprocessing.vad
out = ROOT / "data/interim/tracks"
out.mkdir(parents=True, exist_ok=True)
print(f"vad: min_speech_ms={vc.min_speech_ms} min_silence_ms={vc.min_silence_ms} speech_pad_ms={vc.speech_pad_ms}")

for sid in a.sids:
    wav = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav", sr)
    for thr in a.thresholds or list(vc.thresholds):
        speech = run_vad(wav, cfg, thr)
        loud = mask_non_speech(wav, speech, sr)
        quiet, info = ellie_track(wav, sr, speech, **cfg.preprocessing.ellie_track)
        comb = np.clip(loud + quiet, -1.0, 1.0)
        for name, x in (("loud", loud), ("quiet", quiet), ("combined", comb)):
            sf.write(out / f"{sid}_thr{thr}_{name}.wav", x.astype(np.float32), sr, subtype="PCM_16")
        cov = sum(e - s for s, e in speech) / (len(wav) / sr)
        print(f"{sid} thr={thr}: {len(speech)} VAD 区间, 覆盖 {cov:.0%}; quiet 轨 {info}")
print(f"-> {out}")
