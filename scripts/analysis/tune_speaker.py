"""在网格转写（data/interim/grid）上调说话人判别的加权超参数，对照人工标注 data/interim/gold_speaker/。

指标：Ellie / 受访者的 F1 取平均（macro-F1）。标注里 Ellie 说了但转写里根本没有的句子算 Ellie 的漏检，
所以 VAD 阈值导致的 "Ellie 没转写出来" 也会扣分。工作人员（R）和两可句（X）不计。
留一 session 交叉验证（LOSO）：在两个 session 上选参数，报告在第三个上的分数，看会不会过拟合。
  python scripts/analysis/tune_speaker.py
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from src.audio import load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing.ellie import QuestionBank  # noqa: E402
from src.preprocessing.speaker import score, segment_levels, text_scores  # noqa: E402

SIDS = [308, 309, 676]
THRS = ["auto", "0.12", "0.45", "0.7"]
GRID = dict(text_center=[0.5, 0.6, 0.7], text_scale=[0.05], text_weight=[1.0, 2.0, 4.0],
            level_weight=[0.5, 1.0, 2.0, 4.0, 8.0], level_d0=[1.0, 3.0], level_tau=[0.5],
            track_weight=[0.0, 0.5, 1.0, 2.0], bias=[-1.0, -0.5, 0.0], anchor_score=[0.8],
            question_weight=[0.0, 0.5, 1.0, 2.0], first_person_weight=[0.0, 0.5, 1.0, 2.0])


def align(segs, gold, win=3.0):
    """转写句 -> 标注句：起点相差 <= win 秒的标注句里文本最像的那个（相似度 < 50 记 None）。
    弱声轨上的句级时间戳不可靠（会被拉长），只用起点 + 文本。
    覆盖（标注句有没有被转写出来）：±5 s 内有转写句和它相似度 >= 60；<= 2 词的短句看 ±1.5 s 内有没有转写句开始。"""
    from rapidfuzz import fuzz

    from src.preprocessing.ellie import normalize

    gt = [(x["start"], normalize(x["text"]), x["label"]) for x in gold]
    labs = []
    for s in segs:
        q = normalize(s.get("text", ""))
        cand = [(fuzz.ratio(q, t) if len(q.split()) <= 2 or len(t.split()) <= 2 else fuzz.partial_ratio(q, t), lab)
                for st, t, lab in gt if abs(st - s["start"]) <= win and t]
        best = max(cand, default=(0, None))
        labs.append(best[1] if best[0] >= 50 else None)
    covered = np.zeros(len(gold), bool)
    hyp = [(x["start"], normalize(x.get("text", ""))) for x in segs]
    for i, (st, t, _) in enumerate(gt):
        if len(t.split()) <= 2:
            covered[i] = any(abs(h[0] - st) <= 1.5 for h in hyp)
        else:
            covered[i] = any(fuzz.partial_ratio(t, h[1]) >= 60 for h in hyp if abs(h[0] - st) <= 5 and h[1])
    return labs, covered


def f1_fast(pe, d):
    """pe: bool 数组（判成 Ellie）。d: 预先算好的 isE / isP / miss_e / miss_p。返回 macro-F1。"""
    tp = (pe & d["isE"]).sum()
    fn = (~pe & d["isE"]).sum()
    fp = (pe & d["isP"]).sum()
    tn = (~pe & d["isP"]).sum()
    f_e = 2 * tp / max(2 * tp + fp + fn + d["miss_e"], 1)
    f_p = 2 * tn / max(2 * tn + fn + fp + d["miss_p"], 1)
    return (f_e + f_p) / 2


def macro_f1(pred_e, labs, covered, gold):
    tp = fp = fn = tn = 0
    for pe, gl in zip(pred_e, labs):
        if gl == "E":
            tp, fn = tp + pe, fn + (not pe)
        elif gl == "P":
            fp, tn = fp + pe, tn + (not pe)
    miss_e = sum(1 for c, x in zip(covered, gold) if not c and x["label"] == "E")
    miss_p = sum(1 for c, x in zip(covered, gold) if not c and x["label"] == "P")
    fn_e = fn + miss_e
    f_e = 2 * tp / max(2 * tp + fp + fn_e, 1)
    f_p = 2 * tn / max(2 * tn + fn + fp + miss_p, 1)
    return (f_e + f_p) / 2, {"tp": tp, "fp": fp, "fn": fn, "miss_e": miss_e, "miss_p": miss_p, "f_e": f_e, "f_p": f_p}


def main():
    from src.preprocessing.ellie import normalize
    from src.preprocessing.speaker import load_phrases, text_features

    cfg = load_config()
    bank = QuestionBank(cfg)
    golds = {sid: json.load(open(ROOT / f"data/interim/gold_speaker/{sid}.json")) for sid in SIDS}
    harvest = {sid: {normalize(g["text"]) for g in golds[sid] if g["label"] == "E"} for sid in SIDS}
    generic = load_phrases(bank, ROOT / "resources/ellie_phrases.txt")
    base = {}
    for sid in SIDS:
        wav = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav")
        for thr in THRS:
            tr = json.load(open(ROOT / f"data/interim/grid/{sid}_thr{thr}.json"))
            segs = tr["segments"]
            labs, cov = align(segs, golds[sid])
            labs = np.array([x or "" for x in labs])
            qm, fp = text_features(segs)
            base[sid, thr] = dict(segs=segs, lvl=segment_levels(wav, segs, 16000), qm=qm, fp=fp,
                                  quiet=np.array([x.get("track", "quiet" if x.get("speaker") == "ELLIE" else "loud")
                                                 == "quiet" for x in segs], float),
                                  isE=labs == "E", isP=labs == "P", vthr=tr["threshold"],
                                  miss_e=int(sum(not c and g["label"] == "E" for c, g in zip(cov, golds[sid]))),
                                  miss_p=int(sum(not c and g["label"] == "P" for c, g in zip(cov, golds[sid]))))
    txt_cache = {}

    def txt(sid, thr, setting, sources):
        key = (sid, thr, setting, frozenset(sources))
        if key not in txt_cache:
            ph = generic if setting == "B" else sorted(set(bank.texts).union(*[harvest[x] for x in sources]))
            txt_cache[key] = text_scores(base[sid, thr]["segs"], ph)
        return txt_cache[key]

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} combos; misses (Ellie/participant sentences never transcribed):",
          {f"{s}@{t}": (base[s, t]["miss_e"], base[s, t]["miss_p"]) for s in SIDS for t in THRS})

    def evaluate(sid, thr, setting, sources, cands):
        d = base[sid, thr]
        t = txt(sid, thr, setting, sources)
        return np.array([f1_fast(score(d["lvl"], t, d["quiet"], combos[ci], d["qm"], d["fp"])[0] >= 0.5, d)
                         for ci in cands])

    all_ci = np.arange(len(combos))
    summary = {}
    for setting in ["A", "B"]:
        print(f"\n===== 设定 {setting}：" + ("问题库 + 其他 session 收集的 Ellie 句（干净）" if setting == "A"
                                          else "问题库 + 通用 Ellie 短语表（偏乐观）"))
        for thr in THRS:
            loso, picks = [], []
            for test in SIDS:
                train = [s for s in SIDS if s != test]
                # 训练 session 的短语只来自另一个训练 session，测试 session 的短语来自两个训练 session
                sc_train = np.mean([evaluate(s, thr, setting, [x for x in train if x != s], all_ci)
                                    for s in train], axis=0)
                b = int(np.argmax(sc_train))
                picks.append(b)
                loso.append(evaluate(test, thr, setting, train, [b])[0])
            # 全部 3 个 session 上的最优（每个 session 的短语来自另外两个）
            full = np.mean([evaluate(s, thr, setting, [x for x in SIDS if x != s], all_ci) for s in SIDS], axis=0)
            b = int(np.argmax(full))
            summary[setting, thr] = (np.mean(loso), b)
            print(f"thr={thr:5s} LOSO={np.mean(loso):.3f} [{' '.join(f'{s}:{x:.3f}' for s, x in zip(SIDS, loso))}]"
                  f"  in-sample best={full[b]:.3f}")
        best_thr = max(THRS, key=lambda t: summary[setting, t][0])
        b = summary[setting, best_thr][1]
        print(f"-> 最好的 VAD 阈值策略: {best_thr}；参数: {combos[b]}")
        # 消融：把某个线索关掉（权重=0 或最小），在剩余网格里取最优（in-sample，auto 阈值）
        print("   消融 (thr=auto, in-sample):")
        for name, cond in [("full", lambda c: True),
                           ("-level", lambda c: c["level_weight"] == 0.5 and c["level_d0"] == 3.0),
                           ("-track", lambda c: c["track_weight"] == 0.0),
                           ("-question", lambda c: c["question_weight"] == 0.0),
                           ("-first_person", lambda c: c["first_person_weight"] == 0.0),
                           ("text only", lambda c: c["track_weight"] == 0 and c["question_weight"] == 0
                            and c["first_person_weight"] == 0 and c["level_weight"] == 0.5 and c["level_d0"] == 3.0)]:
            cand = [ci for ci in all_ci if cond(combos[ci])]
            sc = np.mean([evaluate(s, "auto", setting, [x for x in SIDS if x != s], cand) for s in SIDS], axis=0)
            print(f"     {name:14s} {sc.max():.3f}")
        if setting == "A":
            out = ROOT / "data/interim/grid/best_speaker_params.json"
            json.dump({"vad": best_thr, **combos[b]}, open(out, "w"), indent=1)
        for s in SIDS:
            d = base[s, best_thr]
            t = txt(s, best_thr, setting, [x for x in SIDS if x != s])
            p, info = score(d["lvl"], t, d["quiet"], combos[b], d["qm"], d["fp"])
            pe = p >= 0.5
            print(f"   {s}: F1={f1_fast(pe, d):.3f} tp={(pe & d['isE']).sum()} fp={(pe & d['isP']).sum()} "
                  f"fn={(~pe & d['isE']).sum()} miss_e={d['miss_e']}  d'={info['dprime']:.2f} "
                  f"w_level={info['w_level']:.2f} "
                  f"ellie_is_loud={info['ellie_is_loud']} vad_thr={d['vthr']:.2f}")


if __name__ == "__main__":
    main()
