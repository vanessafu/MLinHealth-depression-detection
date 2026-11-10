"""对 data/raw 下的录音跑预处理：vad -> transcribe -> qa -> segments -> wavlm（每步有缓存）。

用法:
  python scripts/preprocessing/run_preprocess.py                       # 全部 session、全部步骤
  python scripts/preprocessing/run_preprocess.py --steps qa segments   # 只重跑后面几步（如改了问题库）
  python scripts/preprocessing/run_preprocess.py --sids 300 301 600 --overwrite
  python scripts/preprocessing/run_preprocess.py --set preprocessing.qa.keep_rapport=true  # A5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402
import traceback  # noqa: E402

import _bootstrap  # noqa: E402,F401
from tqdm import tqdm  # noqa: E402

from src.audio import find_audio  # noqa: E402
from src.config import base_argparser, config_from_args, resolve  # noqa: E402
from src.preprocessing.pipeline import STEPS, process_session  # noqa: E402


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--sids", nargs="*", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    audio = find_audio(cfg)
    if not audio:
        sys.exit(f"在 {resolve(cfg.paths.raw_audio)} 下没找到 {cfg.paths.audio_glob}")
    sids = args.sids or sorted(audio)
    print(f"{len(sids)} sessions, steps={args.steps}")
    log = []
    for sid in tqdm(sids):
        if sid not in audio:
            log.append({"sid": sid, "error": "audio not found"})
            continue
        try:
            log.append(process_session(sid, audio[sid], cfg, args.steps, cfg.experiment.device, args.overwrite))
        except Exception as e:  # 单个 session 失败不影响其他
            traceback.print_exc()
            log.append({"sid": sid, "error": repr(e)})
    errs = [r for r in log if "error" in r]
    out = resolve(cfg.paths.interim) / "preprocess_log.json"
    json.dump(log, open(out, "w"), indent=1)
    print(f"done: {len(log) - len(errs)} ok, {len(errs)} failed -> {out}")


if __name__ == "__main__":
    main()
