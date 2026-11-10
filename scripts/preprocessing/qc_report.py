"""预处理质检：每个 session 的 Q-A 对数、回答时长、片段数、各题是否命中问题映射表。

用法: python scripts/preprocessing/qc_report.py [--sample 10]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402
import random  # noqa: E402

import _bootstrap  # noqa: E402,F401

from src.audio import find_audio  # noqa: E402
from src.config import base_argparser, config_from_args, resolve  # noqa: E402
from src.preprocessing import interim_path  # noqa: E402
from src.preprocessing.qc import qc_report  # noqa: E402


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--sample", type=int, default=10, help="随机抽多少个 session 打印 Q-A 对供人工核对")
    args = ap.parse_args()
    cfg = config_from_args(args)

    sids = sorted(find_audio(cfg))
    df = qc_report(cfg, sids)
    out = resolve(cfg.paths.interim) / "qc_report.csv"
    df.to_csv(out, index=False)
    ok = df[df.status == "ok"]
    print(ok.describe().T[["mean", "min", "50%", "max"]].to_string())
    print("\n各题命中问题映射表的 session 比例:")
    print(ok.filter(like="map_hit_item").mean().round(3).to_string())

    # 问题库里缺的 Ellie 句子，按出现的 session 数排序
    from collections import Counter

    from src.preprocessing.ellie import normalize

    miss = Counter()
    for sid in ok.sid:
        qa = json.load(open(interim_path(cfg, "qa_pairs", sid, "json")))
        miss.update({normalize(t) for t in qa.get("unmatched_ellie", [])})
    print(f"\n问题库未匹配的 Ellie 句子（前 30，共 {len(miss)} 种）——考虑补进 {cfg.preprocessing.ellie.question_bank}:")
    for text, n in miss.most_common(30):
        print(f"  {n:4d}  {text}")

    # 人工核对用：至少一半来自 600 段
    ai = [s for s in ok.sid if 600 <= s <= 718]
    woz = [s for s in ok.sid if s < 600]
    rng = random.Random(0)
    pick = rng.sample(ai, min(len(ai), (args.sample + 1) // 2)) + rng.sample(woz, min(len(woz), args.sample // 2))
    for sid in pick:
        qa = json.load(open(interim_path(cfg, "qa_pairs", sid, "json")))
        print(f"\n===== {sid}  ellie={qa['ellie_speaker']}  counts={qa['speaker_match_counts']}")
        for p in qa["pairs"]:
            print(f"[{p['pair_id']:>2}] {p['t_start']:7.1f}s items={p['items']}  Q: {p['question']}")
            print(f"       A: {p['answer'][:200]}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
