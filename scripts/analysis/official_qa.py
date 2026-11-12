"""试跑 2 的 Q-A 对：Ellie 的问题和时间取自新方法转写（data/interim/transcripts，method=compressed），
受访者的话全部换成 E-DAIC 官方 <sid>_Transcript.csv。合成的转写写到 --config 的 paths.interim/transcripts，
再用管线自己的 qa 步骤（pipeline.process_session）生成 Q-A 对，保证配对逻辑和另外两种完全一样。

官方转写的处理：
  - 只取访谈开始（新方法转写的 interview_start）之后的句子（之前是工作人员调设备）
  - 丢掉像 Ellie 台词的句子：Ellie 响的 session 里 Google ASR 也把她收了进去（308 / 676）
  - 官方只有句级时间，词级时间在句内均分

  python scripts/analysis/official_qa.py --config config/trial/official.yaml --sids 300 425 ...
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import find_audio  # noqa: E402
from src.config import base_argparser, config_from_args, load_config, resolve  # noqa: E402
from src.preprocessing import interim_path  # noqa: E402
from src.preprocessing.ellie import QuestionBank, normalize  # noqa: E402
from src.preprocessing.pipeline import process_session  # noqa: E402
from src.preprocessing.speaker import load_phrases  # noqa: E402


def is_ellie_line(t: str, phrases: list[str]) -> bool:
    from rapidfuzz import fuzz, process

    if len(t.split()) < 3:
        return False
    hit = process.extractOne(t, phrases, scorer=fuzz.ratio)
    return bool(hit and hit[1] >= 80) or any(fuzz.partial_ratio(q, t) >= 90 for q in phrases if len(q.split()) >= 4)


def official_segments(csv: Path, t0: float, phrases: list[str]) -> tuple[list[dict], int]:
    df = pd.read_csv(csv)
    segs, n_ellie = [], 0
    for r in df.itertuples():
        text = str(r.Text).strip()
        if r.Start_Time < t0 - 1 or not normalize(text):
            continue
        if is_ellie_line(normalize(text), phrases):
            n_ellie += 1
            continue
        toks = text.split()
        step = (r.End_Time - r.Start_Time) / len(toks)
        words = [{"word": w, "start": r.Start_Time + i * step, "end": r.Start_Time + (i + 1) * step,
                  "speaker": "PARTICIPANT"} for i, w in enumerate(toks)]
        segs.append({"start": float(r.Start_Time), "end": float(r.End_Time), "text": text, "speaker": "PARTICIPANT",
                     "track": "official", "words": words})
    return segs, n_ellie


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--source", default="config/default.yaml", help="取 Ellie 句子的转写（method=compressed）用的配置")
    ap.add_argument("--sids", nargs="*", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    src_cfg = load_config(args.source)
    audio = find_audio(cfg)
    bank = QuestionBank(cfg)
    phrases = load_phrases(bank, resolve(cfg.preprocessing.speaker.phrases))
    log = []
    for sid in args.sids or sorted(audio):
        src_tr = interim_path(src_cfg, "transcripts", sid, "json")
        csv = audio[sid].parent / f"{sid}_Transcript.csv"
        if not src_tr.exists() or not csv.exists():
            log.append({"sid": sid, "error": f"missing {src_tr if not src_tr.exists() else csv}"})
            continue
        tr = json.load(open(src_tr))
        ellie = [s for s in tr["segments"] if s.get("speaker") == "ELLIE"]
        part, n_ellie = official_segments(csv, tr["interview_start"], phrases)
        out = {"segments": sorted(ellie + part, key=lambda s: s["start"]), "interview_start": tr["interview_start"],
               "method": "official_answers", "n_official_dropped_as_ellie": n_ellie}
        p_tr = interim_path(cfg, "transcripts", sid, "json")
        if p_tr.exists() and not args.overwrite:
            pass
        else:
            json.dump(out, open(p_tr, "w"), ensure_ascii=False, indent=1)
        # qa 步骤要读 vad/<sid>.json 里的 duration：从来源目录拷过来
        p_vad = interim_path(cfg, "vad", sid, "json")
        if not p_vad.exists():
            shutil.copy(interim_path(src_cfg, "vad", sid, "json"), p_vad)
        st = process_session(sid, audio[sid], cfg, ["qa"], cfg.experiment.device, args.overwrite)
        st.update({"n_ellie_segs": len(ellie), "n_official_segs": len(part), "n_official_dropped_as_ellie": n_ellie})
        log.append(st)
        print(json.dumps(st), flush=True)
    print(f"done: {sum('ok' in r for r in log)} ok, {sum('error' in r for r in log)} failed")


if __name__ == "__main__":
    main()
