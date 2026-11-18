"""不重新转写，只在已有转写上重跑说话人判别（改了 speaker.py / 配置之后用）。

  python scripts/analysis/reassign_speakers.py --sids 334 344 402          # 写回 data/interim/transcripts/
  python scripts/analysis/reassign_speakers.py --sids 308 309 676 --dry-run   # 只看判别结果变没变
写回时保留原文件里的其他字段；访谈开始重新判定（只会往后删，不会恢复已删掉的句子）。
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import find_audio, load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing import interim_path  # noqa: E402
from src.preprocessing.ellie import QuestionBank  # noqa: E402
from src.preprocessing.speaker import assign_speakers, interview_start  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sids", nargs="+", type=int, required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
cfg = load_config()
bank, audio, sr = QuestionBank(cfg), find_audio(cfg), cfg.preprocessing.sample_rate
for sid in a.sids:
    p = interim_path(cfg, "transcripts", sid, "json")
    tr = json.load(open(p))
    old = [s["speaker"] for s in tr["segments"]]
    segs, info = assign_speakers(copy.deepcopy(tr["segments"]), load_wav(audio[sid], sr), bank, cfg)
    t0 = interview_start(segs, bank, cfg.preprocessing.speaker.opening_min_score)
    changed = int(np.sum([o != s["speaker"] for o, s in zip(old, segs)]))
    print(f"{sid}: ellie_is_loud {tr['speaker'].get('ellie_is_loud')} -> {info['ellie_is_loud']}  "
          f"anchors {tr['speaker'].get('n_anchor')} -> {info['n_anchor']}  changed {changed}/{len(segs)}  "
          f"start {tr['interview_start']:.1f} -> {t0:.1f}")
    if not a.dry_run:
        tr.update(segments=[s for s in segs if s["end"] > t0 - 0.5], interview_start=t0, speaker=info)
        json.dump(tr, open(p, "w"), indent=1)
