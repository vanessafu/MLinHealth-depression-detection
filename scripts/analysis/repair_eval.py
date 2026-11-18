"""评估转写修复：before（WhisperX 流程）vs after（官方转写修复，src/preprocessing/repair.py）。

  python scripts/analysis/repair_eval.py --steps metrics accuracy checksheet umap
输出 data/interim/repair_eval/:
  per_session.csv        每个 session × {before, after} 的指标
  report.md / report.json 汇总（全部 / WoZ 300–492 / AI 600–718）
  check_sheet.md + clips/ 5 个随机 session（>= 3 个 600 段）前后对照 + 每个 Ellie 片段的音频
  umap.png / flagged_turns.csv  句向量 UMAP / PCA，标出落在 Ellie 簇里的 "受访者" 句
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.audio import find_audio, load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing import interim_path  # noqa: E402
from src.preprocessing.ellie import QuestionBank, classify_ellie_sentences, normalize, split_sentences  # noqa: E402
from src.preprocessing.qa_pairs import build_qa_pairs  # noqa: E402
from src.preprocessing.repair import (  # noqa: E402  # noqa: E402
    ELLIE,
    PART,
    UNK,
    _EllieLike,
    load_official,
    official_transcript_path,
)
from src.preprocessing.transcribe import words_to_utterances  # noqa: E402

OUT = ROOT / "data/interim/repair_eval"


def group(sid: int) -> str:
    return "AI" if sid >= 600 else "WoZ"


# ------------------------------------------------------------------ 载入
def load_session(cfg, sid, audio):
    p_b, p_a = interim_path(cfg, "transcripts", sid, "json"), interim_path(cfg, "transcripts_repaired", sid, "json")
    if not (p_b.exists() and p_a.exists()):
        return None
    before = json.load(open(p_b))
    after = json.load(open(p_a))
    b_turns = [{**u, "source": "whisperx"} for u in
               words_to_utterances(before, cfg.preprocessing.whisperx.utterance_gap_sec)]
    off_p = official_transcript_path(audio[sid], sid)
    return {"sid": sid, "before": before, "after": after, "b_turns": b_turns, "a_turns": after["turns"],
            "official": load_official(off_p) if off_p else []}


# ------------------------------------------------------------------ 指标
def attach_stats(utts, bank, cfg):
    """mirror build_qa_pairs（保留寒暄题），记下哪些受访者句子挂到了某个问题上。"""
    qc = cfg.preprocessing.qa
    utts = [{**u, "uid": i} for i, u in enumerate(utts)]
    events = classify_ellie_sentences(utts, ELLIE, bank, cfg)
    start = next((i for i, e in enumerate(events) if e["role"] == "ellie" and e["type"] == "main"),
                 len(events)) if qc.trim_opening else 0
    end = len(events)
    if qc.trim_closing:
        last_main = max((i for i, e in enumerate(events) if e["role"] == "ellie" and e["type"] == "main"),
                        default=start)
        end = next((i for i, e in enumerate(events) if i > last_main and e["role"] == "ellie"
                    and e["type"] == "closing"), len(events))
    cur, attached = None, set()
    for e in events[start:end]:
        if e["role"] == "ellie":
            if e["type"] == "main":
                cur = True
        elif cur is not None:
            attached.add(e["uid"])
    part = [u for u in utts if u["speaker"] == PART]
    n_pairs = len(build_qa_pairs(events, cfg)[0])
    return part, attached, n_pairs


def session_metrics(S, bank, cfg):
    rows = []
    for cond, turns in (("before", S["b_turns"]), ("after", S["a_turns"])):
        utts = [t for t in turns if t["speaker"] != UNK]
        part, attached, n_pairs = attach_stats(utts, bank, cfg)
        n_part = len(part)
        short_orphan = sum(len(u["text"].split()) < 3 and u["uid"] not in attached for u in part)
        ellie_turns = [t for t in turns if t["speaker"] == ELLIE]
        q_by_src, main_by_src = Counter(), Counter()
        for t in ellie_turns:
            for s in split_sentences(t["text"]) or [t["text"]]:
                entry, _ = bank.match(s)
                typ = entry.type if entry else cfg.preprocessing.ellie.unmatched_as
                if typ in ("main", "followup"):
                    q_by_src[t["source"]] += 1
                if typ == "main":
                    main_by_src[t["source"]] += 1
        r = {"sid": S["sid"], "group": group(S["sid"]), "cond": cond, "n_turns": len(turns),
             "n_participant": n_part, "pct_answers_with_question": 100 * len(attached) / max(n_part, 1),
             "pct_orphan_short": 100 * short_orphan / max(n_part, 1),
             "pct_unknown": 100 * sum(t["speaker"] == UNK for t in turns) / max(len(turns), 1),
             "n_ellie_turns": len(ellie_turns), "n_qa_pairs": n_pairs,
             "n_questions": sum(q_by_src.values()), "n_main_questions": sum(main_by_src.values())}
        for k, v in q_by_src.items():
            r[f"q_{k}"] = v
        for k, v in main_by_src.items():
            r[f"main_{k}"] = v
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 说话人准确率
def official_part_iv(S, ellie_like, cfg):
    rc = cfg.preprocessing.repair
    return np.array([(o["start"], o["end"]) for o in S["official"]
                     if ellie_like(o["text"]) < rc.ellie_like_score]).reshape(-1, 2)


def word_prf(S, iv, tol=0.3):
    """before 的 WhisperX 词：预测 PARTICIPANT vs 参考 "词中点在官方受访者句子里"（±tol）。"""
    tp = fp = fn = 0
    for seg in S["before"]["segments"]:
        for w in seg.get("words", []):
            if w.get("start") is None:
                continue
            m = (w["start"] + w["end"]) / 2
            ref = bool(len(iv)) and bool(((iv[:, 0] - tol <= m) & (m <= iv[:, 1] + tol)).any())
            pred = w.get("speaker", seg.get("speaker")) == PART
            tp += pred and ref
            fp += pred and not ref
            fn += ref and not pred
    return tp, fp, fn


def time_prf(turns, iv, dur):
    """按 10 ms 网格：预测 "受访者" 的时间 vs 官方受访者句子的时间。"""
    g = np.zeros(int(dur * 100) + 1, bool)
    for s, e in iv:
        g[int(max(s, 0) * 100): int(e * 100)] = True
    p = np.zeros_like(g)
    for t in turns:
        if t["speaker"] == PART:
            p[int(max(t["start"], 0) * 100): int(t["end"] * 100)] = True
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    return int(tp), int(fp), int(fn)


def gold_eval(turns, gold):
    """手工标注（E / P / R / X）：每个标注句找 ±3 s 内文本最像的转写句（相似度 >= 50），看标签对不对。"""
    from rapidfuzz import fuzz

    res = Counter()
    for g in gold:
        if g["label"] not in ("E", "P"):
            continue
        q = normalize(g["text"])
        best, lab = 0, None
        for t in turns:
            if abs(t["start"] - g["start"]) > 3 or not q:
                continue
            h = normalize(t["text_asr"] if t.get("text_asr") else t["text"])
            sc = fuzz.ratio(q, h) if len(q.split()) <= 2 or len(h.split()) <= 2 else fuzz.partial_ratio(q, h)
            if sc > best:
                best, lab = sc, t["speaker"]
        gl = ELLIE if g["label"] == "E" else PART
        if best < 50:
            res[f"{g['label']}_missing"] += 1
        elif lab == gl:
            res[f"{g['label']}_correct"] += 1
        elif lab == UNK:
            res[f"{g['label']}_unknown"] += 1
        else:
            res[f"{g['label']}_wrong"] += 1
    return dict(res)


def daicwoz_eval(S, path, bank):
    """DAIC-WOZ 的 Ellie 原文（如果有）：召回 + 文本准确度。先用高置信匹配估计两边时间轴的偏移。"""
    from rapidfuzz import fuzz

    d = pd.read_csv(path, sep="\t")
    ell = [(float(r.start_time), normalize(str(r.value))) for r in d.itertuples() if str(r.speaker) == "Ellie"]
    ell = [(t, x) for t, x in ell if len(x.split()) >= 3]
    hyp = [(t["start"], normalize(t["text"])) for t in S["turns_for_daic"] if t["speaker"] == ELLIE]
    pairs = [(h[0] - e[0]) for e in ell for h in hyp if fuzz.ratio(e[1], h[1]) >= 90]
    off = float(np.median(pairs)) if pairs else 0.0
    rec, sims = 0, []
    for t, x in ell:
        c = [fuzz.ratio(x, h[1]) for h in hyp if abs(h[0] - off - t) <= 5]
        if c and max(c) >= 60:
            rec += 1
            sims.append(max(c))
    return {"n_ellie": len(ell), "recalled": rec, "mean_text_sim": float(np.mean(sims)) if sims else float("nan"),
            "offset_s": off}


# ------------------------------------------------------------------ 对照表
def checksheet(sessions, cfg, audio, n=5, n_ai=3, seed=0):
    import soundfile as sf

    rng = random.Random(seed)
    ai = [s for s in sessions if s >= 600]
    woz = [s for s in sessions if s < 600]
    pick = sorted(rng.sample(ai, min(n_ai, len(ai))) + rng.sample(woz, n - min(n_ai, len(ai))))
    clip_dir = OUT / "clips"
    md = ["# Repair check sheet", "",
          f"Sessions (seed {seed}): {', '.join(map(str, pick))}. Rows = overlapping time blocks; "
          "left = before (WhisperX pipeline), right = after (official-transcript repair). "
          "Clips are the original audio of the block ±0.3 s, loudness-normalised so quiet Ellie is audible.", ""]
    rows = []
    for sid in pick:
        S = load_session(cfg, sid, audio)
        wav = load_wav(audio[sid], cfg.preprocessing.sample_rate)
        sr = cfg.preprocessing.sample_rate
        items = [("B", t) for t in S["b_turns"]] + [("A", t) for t in S["a_turns"]]
        items.sort(key=lambda x: x[1]["start"])
        blocks, cur, cur_end = [], [], -1
        for side, t in items:
            if cur and t["start"] > cur_end:
                blocks.append(cur)
                cur = []
            cur.append((side, t))
            cur_end = max(cur_end, t["end"]) if len(cur) > 1 else t["end"]
        if cur:
            blocks.append(cur)
        md += [f"## {sid} ({group(sid)})", "",
               "| # | time | before | after | clip |", "|---|---|---|---|---|"]
        for k, bl in enumerate(blocks):
            s0, e0 = min(t["start"] for _, t in bl), max(t["end"] for _, t in bl)

            def cell(side, bl=bl):
                return "<br>".join(f"**{t['speaker'][:1]}**{'' if side == 'B' else '·' + t['source']} "
                                   f"{t['text'].replace('|', '/')[:160]}" for sd, t in bl if sd == side)

            need = any(t["speaker"] != PART for _, t in bl)
            clip = ""
            if need:
                a, b = max(0, int((s0 - 0.3) * sr)), min(len(wav), int((e0 + 0.3) * sr))
                x = wav[a:b].astype(np.float32)
                rms = np.sqrt(np.mean(x ** 2)) + 1e-9
                x = np.clip(x * min(10 ** (-20 / 20) / rms, 10 ** (60 / 20)), -0.99, 0.99)
                cp = clip_dir / str(sid) / f"{k:03d}_{s0:07.1f}.wav"
                cp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(cp, x, sr, subtype="PCM_16")
                clip = str(cp.relative_to(ROOT))
            md.append(f"| {k} | {s0:.1f}–{e0:.1f} | {cell('B')} | {cell('A')} | {clip} |")
            rows.append({"sid": sid, "block": k, "start": round(s0, 2), "end": round(e0, 2),
                         "before": " || ".join(f"[{t['speaker']}] {t['text']}" for sd, t in bl if sd == "B"),
                         "after": " || ".join(f"[{t['speaker']}:{t['source']}] {t['text']}"
                                              for sd, t in bl if sd == "A"),
                         "clip": clip, "verdict": ""})
        md.append("")
    (OUT / "check_sheet.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(OUT / "check_sheet.csv", index=False)
    return pick


# ------------------------------------------------------------------ UMAP
def umap_plot(all_sessions, cfg, audio, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    recs = []
    for sid in all_sessions:
        S = load_session(cfg, sid, audio)
        for cond, turns in (("before", S["b_turns"]), ("after", S["a_turns"])):
            for t in turns:
                txt = t["text"].strip()
                if len(txt.split()) >= 1:
                    recs.append({"sid": sid, "cond": cond, "speaker": t["speaker"], "source": t["source"],
                                 "start": round(t["start"], 2), "text": txt})
    df = pd.DataFrame(recs)
    enc = SentenceTransformer(cfg.stage1.retrieval.encoder, device=device)
    X = enc.encode(df.text.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    # Ellie 簇：同一条件下 15 个最近邻里 >= 70% 是 ELLIE 的受访者句
    df["flag"] = False
    for cond in ("before", "after"):
        m = (df.cond == cond).to_numpy()
        nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X[m])
        _, idx = nn.kneighbors(X[m])
        lab = df.speaker[m].to_numpy()
        frac_e = (lab[idx[:, 1:]] == ELLIE).mean(1)
        df.loc[m, "frac_ellie_nn"] = frac_e
        df.loc[m, "flag"] = (lab == PART) & (frac_e >= 0.7)
    Xp = PCA(n_components=50, random_state=0).fit_transform(X)
    U = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=0).fit_transform(Xp)
    P2 = Xp[:, :2]
    df["umap_x"], df["umap_y"], df["pca_x"], df["pca_y"] = U[:, 0], U[:, 1], P2[:, 0], P2[:, 1]
    col = {ELLIE: "#d62728", PART: "#1f77b4", UNK: "#7f7f7f"}
    fig, axs = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for j, cond in enumerate(("before", "after")):
        d = df[df.cond == cond]
        for i, (xs, ys, name) in enumerate((("umap_x", "umap_y", "UMAP"), ("pca_x", "pca_y", "PCA"))):
            ax = axs[i, j]
            for spk in (PART, UNK, ELLIE):
                e = d[d.speaker == spk]
                ax.scatter(e[xs], e[ys], s=1.5, c=col[spk], alpha=0.35, label=f"{spk} ({len(e)})", rasterized=True)
            f = d[d.flag]
            ax.scatter(f[xs], f[ys], s=10, marker="x", c="k", lw=0.6,
                       label=f"PARTICIPANT inside Ellie cluster ({len(f)})")
            ax.set_title(f"{name} — {cond} ({len(d)} turns)")
            ax.legend(markerscale=4, loc="upper right", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("bge-large-en-v1.5 embeddings of all turns (UMAP fitted jointly on before + after)")
    fig.savefig(OUT / "umap.png", dpi=110)
    df[df.flag].sort_values(["cond", "sid", "start"])[
        ["cond", "sid", "start", "source", "frac_ellie_nn", "text"]].to_csv(OUT / "flagged_turns.csv", index=False)
    return df.groupby("cond").flag.sum().to_dict(), df.groupby(["cond", "speaker"]).size().to_dict()


# ------------------------------------------------------------------ 汇总
def summarize(per, extra):
    def agg(d):
        out = {}
        for cond in ("before", "after"):
            x = d[d.cond == cond]
            if x.empty:
                continue
            # 百分比按句子数加权（= 把所有 session 的句子合在一起算）
            o = {"n_sessions": len(x),
                 "pct_answers_with_question": (x.pct_answers_with_question * x.n_participant).sum()
                 / x.n_participant.sum(),
                 "pct_orphan_short": (x.pct_orphan_short * x.n_participant).sum() / x.n_participant.sum(),
                 "pct_unknown": (x.pct_unknown * x.n_turns).sum() / x.n_turns.sum(),
                 "ellie_turns": x.n_ellie_turns.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_dict(),
                 "qa_pairs": x.n_qa_pairs.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_dict(),
                 "sessions_le_3_pairs": int((x.n_qa_pairs <= 3).sum()),
                 "questions_by_source": {c[2:]: int(x[c].sum()) for c in x.columns if c.startswith("q_")
                                         and x[c].sum() > 0},
                 "main_questions_by_source": {c[5:]: int(x[c].sum()) for c in x.columns if c.startswith("main_")
                                              and x[c].sum() > 0}}
            out[cond] = o
        return out

    rep = {"overall": agg(per)}
    for g in ("WoZ", "AI"):
        rep[g] = agg(per[per.group == g])
    rep.update(extra)
    return rep


def write_md(rep):
    L = ["# Transcript repair: before vs after", ""]
    for scope in ("overall", "WoZ", "AI"):
        r = rep[scope]
        if "before" not in r or "after" not in r:
            continue
        L += [f"## {scope} (n = {r['before']['n_sessions']} sessions)", "",
              "| metric | before | after |", "|---|---|---|"]
        for k, name in (("pct_answers_with_question", "% participant answers with an Ellie question attached"),
                        ("pct_orphan_short", "% orphaned short answers (<3 words, no question)"),
                        ("pct_unknown", "% turns labeled unknown")):
            L.append(f"| {name} | {r['before'][k]:.1f} | {r['after'][k]:.1f} |")
        for k, name in (("ellie_turns", "Ellie turns / session"), ("qa_pairs", "Q-A pairs / session")):
            fmt = lambda d: f"median {d['50%']:.0f} (IQR {d['25%']:.0f}–{d['75%']:.0f}, min {d['min']:.0f}, max {d['max']:.0f})"  # noqa: E731,E501
            L.append(f"| {name} | {fmt(r['before'][k])} | {fmt(r['after'][k])} |")
        L.append(f"| sessions with ≤ 3 Q-A pairs | {r['before']['sessions_le_3_pairs']} | "
                 f"{r['after']['sessions_le_3_pairs']} |")
        L.append(f"| Ellie questions (main + follow-up) by source | {r['before']['questions_by_source']} | "
                 f"{r['after']['questions_by_source']} |")
        L.append(f"| …of which bank-matched main questions | {r['before']['main_questions_by_source']} | "
                 f"{r['after']['main_questions_by_source']} |")
        L.append("")
    for k in ("speaker_vs_official", "gold", "daicwoz", "umap"):
        if k in rep:
            L += [f"## {k}", "", "```", json.dumps(rep[k], indent=1, default=float), "```", ""]
    (OUT / "report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=["metrics", "accuracy", "checksheet", "umap"])
    ap.add_argument("--sid-file", default=str(ROOT / "data/interim/repair_sids.txt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--daicwoz-dir", default=None, help="DAIC-WOZ XXX_TRANSCRIPT.csv 所在目录（可选）")
    a = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    audio = find_audio(cfg)
    bank = QuestionBank(cfg)
    sids = [int(x) for x in open(a.sid_file).read().split()]
    have = [s for s in sids if interim_path(cfg, "transcripts_repaired", s, "json").exists()]
    print(f"{len(have)}/{len(sids)} sessions have before + after transcripts", flush=True)
    split_ids = set()
    for f in ("train_split.csv", "dev_split.csv"):
        p = ROOT / "data/labels" / f
        if p.exists():
            split_ids |= set(pd.read_csv(p).Participant_ID.astype(int))
    stat_sids = [s for s in have if s in split_ids]          # 统计只用 train + dev（676 只用于手工标注）

    extra = {}
    if (OUT / "extra.json").exists():
        extra = json.load(open(OUT / "extra.json"))
    if "metrics" in a.steps:
        rows = []
        for sid in stat_sids:
            rows += session_metrics(load_session(cfg, sid, audio), bank, cfg)
        per = pd.DataFrame(rows).fillna(0)
        per.to_csv(OUT / "per_session.csv", index=False)
    if "accuracy" in a.steps:
        el = _EllieLike(bank, cfg)
        acc = {"before_words": Counter(), "before_time": Counter(), "after_time": Counter()}
        by_g = {g: {"before_words": Counter()} for g in ("WoZ", "AI")}
        for sid in stat_sids:
            S = load_session(cfg, sid, audio)
            iv = official_part_iv(S, el, cfg)
            tp, fp, fn = word_prf(S, iv)
            for c, v in zip("tp fp fn".split(), (tp, fp, fn)):
                acc["before_words"][c] += v
                by_g[group(sid)]["before_words"][c] += v
            dur = max([t["end"] for t in S["b_turns"] + S["a_turns"]] + [0]) + 1
            for cond, turns in (("before_time", S["b_turns"]), ("after_time", S["a_turns"])):
                for c, v in zip("tp fp fn".split(), time_prf(turns, iv, dur)):
                    acc[cond][c] += v

        def prf(c):
            p = c["tp"] / max(c["tp"] + c["fp"], 1)
            r = c["tp"] / max(c["tp"] + c["fn"], 1)
            return {"precision": round(p, 4), "recall": round(r, 4), **{k: int(v) for k, v in c.items()}}

        extra["speaker_vs_official"] = {
            "note": "reference = official participant sentences (Ellie-like official lines removed), ±0.3 s. "
                    "after_time is NOT independent: after takes participant turns from the official transcript.",
            **{k: prf(v) for k, v in acc.items()},
            **{f"before_words_{g}": prf(v["before_words"]) for g, v in by_g.items()}}
        gold = {}
        for sid in (308, 309, 676):
            gp = ROOT / f"data/interim/gold_speaker/{sid}.json"
            if gp.exists() and sid in have:
                S = load_session(cfg, sid, audio)
                g = json.load(open(gp))
                gold[sid] = {"before": gold_eval(S["b_turns"], g), "after": gold_eval(S["a_turns"], g)}
        extra["gold"] = gold
        ddir = Path(a.daicwoz_dir) if a.daicwoz_dir else None
        if ddir and ddir.exists():
            res = {"before": Counter(), "after": Counter()}
            for sid in stat_sids:
                f = ddir / f"{sid}_TRANSCRIPT.csv"
                if sid >= 600 or not f.exists():
                    continue
                S = load_session(cfg, sid, audio)
                for cond, turns in (("before", S["b_turns"]), ("after", S["a_turns"])):
                    S["turns_for_daic"] = turns
                    r = daicwoz_eval(S, f, bank)
                    res[cond]["n_ellie"] += r["n_ellie"]
                    res[cond]["recalled"] += r["recalled"]
            extra["daicwoz"] = {k: {**dict(v), "recall": v["recalled"] / max(v["n_ellie"], 1)}
                                for k, v in res.items()}
        else:
            extra["daicwoz"] = "DAIC-WOZ Ellie transcripts not available (pass --daicwoz-dir); skipped"
    if "checksheet" in a.steps:
        extra["checksheet_sessions"] = checksheet(stat_sids, cfg, audio)
    if "umap" in a.steps:
        flags, sizes = umap_plot(stat_sids, cfg, audio, a.device)
        extra["umap"] = {"participant_turns_in_ellie_cluster": flags,
                         "turns": {f"{k[0]}:{k[1]}": int(v) for k, v in sizes.items()}}
    json.dump(extra, open(OUT / "extra.json", "w"), indent=1, default=float)
    per = pd.read_csv(OUT / "per_session.csv").fillna(0)
    rep = summarize(per, extra)
    json.dump(rep, open(OUT / "report.json", "w"), indent=1, default=float)
    write_md(rep)
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
