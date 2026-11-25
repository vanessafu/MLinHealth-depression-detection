"""从 train 的 Ellie 句子里挖问题库（不依赖旧问题库；旧库只用来继承题型 / PHQ 题号）。

来源：data/interim/transcripts_repaired/ 里 train session 的 ELLIE 句（text_asr，即原始 ASR 文本），
      这些 Ellie 标签来自 "官方受访者转写的空隙"，不是旧问题库的文本匹配，所以能挖到库外的问题。
步骤：
  1. 按句切分、normalize；>= 3 词的句子用 fuzz.ratio >= --sim 贪心聚类（按出现次数从高到低），1–2 词只合并完全相同的
  2. 一个簇在 >= --min-sessions 个不同 session 里出现才保留（Ellie 的话是固定句式，会跨 session 重复；
     Whisper 编的句子、零星的受访者话不会）
  3. 同一簇在官方受访者转写里也常出现（受访者 session 数 >= Ellie session 数的一半）-> 丢掉（"i don't know"）；
     纯语气词（um / uh / hmm / mm hmm）也丢：说话人判别只看文本，进库会把受访者的 "um" 判成 Ellie
  4. 题型：和旧库某条相似度 >= --sim 就算旧库已有；和旧库开场白 partial_ratio >= 90 -> opening；否则按问句形式推断
       问句 + 追问用语（that / more / why / example / how so ...）且 <= 8 词 -> followup；其他问句 -> main；
       非问句 <= 4 词 -> backchannel；含 goodbye / asked everything -> closing；其余 -> followup
     PHQ 题号：新 main 问题和旧库 main 问题 fuzz.ratio >= 80 就继承它的 items
  5. 输出 = 旧库 ∪ 挖到的新句（新句 notes 列写 "mined:<session 数>"）

  python scripts/analysis/mine_ellie_bank.py --out resources/ellie_questions_mined.csv
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.ellie import normalize, split_sentences  # noqa: E402

FOLLOW = re.compile(r"\b(that|tell me more|more about|why|example|how so|what happened|like what|how come|"
                    r"anything else)\b")
FILLER_TOKENS = {"um", "uh", "hmm", "mm", "mmm", "hm", "huh", "mhm", "ah", "oh", "uh-huh", "mm-hmm", "er"}
FILLERS = {"um", "uh", "hmm", "mm", "mm hmm", "uh huh", "mhm", "hm", "ah", "oh", "um um", "uh uh"}
QSTART = re.compile(r"^(what|how|why|when|where|who|which|do|does|did|are|is|was|were|have|has|had|can|could|"
                    r"would|will|tell me|so how|so what|and what|and how)\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "resources/ellie_questions_mined.csv"))
    ap.add_argument("--old", default=str(ROOT / "resources/ellie_questions.csv"))
    ap.add_argument("--sim", type=float, default=85)
    ap.add_argument("--min-sessions", type=int, default=3)
    a = ap.parse_args()
    from rapidfuzz import fuzz, process

    train = set(pd.read_csv(ROOT / "data/labels/train_split.csv").Participant_ID.astype(int))
    occ = defaultdict(set)          # normalized sentence -> sessions
    qmark = Counter()               # normalized sentence -> 以 ? 结尾的次数
    raw_n = Counter()
    for sid in sorted(train):
        p = ROOT / f"data/interim/transcripts_repaired/{sid}.json"
        if not p.exists():
            continue
        for t in json.load(open(p))["turns"]:
            if t["speaker"] != "ELLIE":
                continue
            for s in split_sentences(t.get("text_asr") or t["text"]) or [t["text"]]:
                q = normalize(s)
                if not q:
                    continue
                occ[q].add(sid)
                raw_n[q] += 1
                qmark[q] += s.strip().endswith("?")
    print(f"{sum(raw_n.values())} Ellie sentences, {len(occ)} unique (normalized)")

    # 1. 贪心聚类
    order = sorted(occ, key=lambda q: (-len(occ[q]), -raw_n[q]))
    reps, members = [], []
    for q in order:
        j = None
        if len(q.split()) >= 3 and reps:
            hit = process.extractOne(q, reps, scorer=fuzz.ratio, score_cutoff=a.sim)
            if hit:
                j = hit[2]
        elif q in reps:
            j = reps.index(q)
        if j is None:
            reps.append(q)
            members.append([q])
        else:
            members[j].append(q)
    clusters = []
    for rep, mem in zip(reps, members):
        sess = set().union(*(occ[m] for m in mem))
        clusters.append({"rep": rep, "n_sessions": len(sess), "n": sum(raw_n[m] for m in mem),
                         "qfrac": sum(qmark[m] for m in mem) / max(sum(raw_n[m] for m in mem), 1),
                         "n_variants": len(mem)})
    cl = [c for c in clusters if c["n_sessions"] >= a.min_sessions]
    print(f"{len(clusters)} clusters, {len(cl)} in >= {a.min_sessions} sessions")

    # 3. 官方受访者转写里常见的 -> 丢
    part = defaultdict(set)
    off_texts = []
    for sid in sorted(train):
        f = next(iter((ROOT / "data/raw").glob(f"*/{sid}_P/{sid}_Transcript.csv")), None)
        if f is None:
            continue
        d = pd.read_csv(f)
        for t in d.Text.dropna().astype(str):
            off_texts.append((sid, normalize(t)))
    off_norm = [t for _, t in off_texts]
    for c in cl:
        for _, _sc, i in process.extract(c["rep"], off_norm, scorer=fuzz.ratio, score_cutoff=a.sim, limit=None):
            part[c["rep"]].add(off_texts[i][0])
    kept, dropped = [], []
    for c in cl:
        c["n_part_sessions"] = len(part[c["rep"]])
        filler = all(t in FILLER_TOKENS for t in c["rep"].split())
        bad = c["n_part_sessions"] >= max(2, c["n_sessions"] / 2) or c["rep"] in FILLERS or filler
        (dropped if bad else kept).append(c)
    print(f"dropped {len(dropped)} also common in official participant text, e.g.",
          [d["rep"] for d in sorted(dropped, key=lambda x: -x["n_sessions"])[:12]])

    # 4. 题型
    old = pd.read_csv(a.old, dtype=str).fillna("")
    old["norm"] = old.text.map(normalize)
    opening = old.norm[old.type == "opening"].tolist()
    mains = old[(old.type == "main") & (old["items"] != "")]
    rows, n_inherit = [], 0
    for c in sorted(kept, key=lambda x: -x["n_sessions"]):
        hit = process.extractOne(c["rep"], old.norm.tolist(), scorer=fuzz.ratio, score_cutoff=a.sim)
        if hit:
            n_inherit += 1
            continue                                   # 旧库已有
        w = c["rep"].split()
        is_q = c["qfrac"] >= 0.5 or bool(QSTART.match(c["rep"]))
        items = ""
        if max((fuzz.partial_ratio(c["rep"], o) for o in opening), default=0) >= 90 and len(w) >= 3:
            typ = "opening"
        elif "goodbye" in c["rep"] or "asked everything" in c["rep"]:
            typ = "closing"
        elif is_q and FOLLOW.search(c["rep"]) and len(w) <= 8:
            typ = "followup"
        elif is_q and len(w) >= 3:
            typ = "main"
        elif len(w) <= 4:
            typ = "backchannel"
        else:
            typ = "followup"
        if typ == "main" and len(mains):
            hit = process.extractOne(c["rep"], mains.norm.tolist(), scorer=fuzz.ratio, score_cutoff=80)
            if hit:
                items = mains.iloc[hit[2]]["items"]
        rows.append({"text": c["rep"], "type": typ, "rapport": "0", "items": items,
                     "notes": f"mined:{c['n_sessions']}"})
    new = pd.DataFrame(rows)
    out = pd.concat([old.drop(columns="norm"), new], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"old bank {len(old)} (+{n_inherit} mined clusters already in it) + new {len(new)} "
          f"({new.type.value_counts().to_dict()}) -> {a.out}")
    print(new.to_string())


if __name__ == "__main__":
    main()
