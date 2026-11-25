"""Stage 1 dev benchmark：旧 / 新（挖掘的）Ellie 问题库 × 检索模型，候选相关性 + 零样本 + LoRA。

中间产物（Q-A 对、检索结果、LoRA 权重）全写在 --work（默认 $TMPDIR，作业结束自动删除），只留 --out 一个 JSON。
  R_old_bge   旧问题库 + bge-large-en-v1.5（= 现在的流程）
  R_new_bge   挖掘的问题库（resources/ellie_questions_mined.csv）+ bge
  R_new_qwen  挖掘的问题库 + Qwen3-Embedding-0.6B
  ALL_new     挖掘的问题库，不检索：每题都放全部 Q-A 对（按时间）
步骤：
  1. qa       旧库：现有 WhisperX 转写直接配对；新库：先用新库重判说话人（只看文本 + 一点响度），再配对
  2. retrieve 3 种检索配置
  3. judge    候选相关性（无标签）：Qwen3-4B-Instruct 判 "这段问答对给这道题打分有没有用"，P(yes)
  4. zeroshot dev 零样本打分（4 种配置）
  5. lora     在 train 上训练 LoRA（train 内 20% 做早停），dev 上评估；检索配置按 judge 的相关性选（不看 dev 标签）

  python scripts/evaluation/stage1_bench.py --out logs/stage1_bench.json
"""
import argparse
import copy
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import N_ITEMS  # noqa: E402
from src.audio import find_audio, load_wav  # noqa: E402
from src.config import load_config  # noqa: E402
from src.labels import load_labels  # noqa: E402

_spec = importlib.util.spec_from_file_location("dzs", ROOT / "scripts/evaluation/dev_zeroshot.py")
dzs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dzs)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_EMB = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTR = ("Instruct: Given a PHQ-8 depression symptom, retrieve interview question-answer exchanges in which "
              "the participant talks about this symptom\nQuery:")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_cfg(work: Path, name: str, bank: str, extra: list[str] = ()):
    return load_config(str(ROOT / "config/default.yaml"), [
        f"paths.interim={work / ('interim_' + name)}", f"paths.outputs={work / 'outputs'}",
        f"preprocessing.ellie.question_bank={bank}", f"stage1.model.base={MODEL}",
        "experiment.device=cuda", f"experiment.name={name}", *extra])


# ------------------------------------------------------------------ 1. qa
def build_qa(cfg, sids, audio, reassign: bool):
    from src.preprocessing import interim_path
    from src.preprocessing.ellie import QuestionBank
    from src.preprocessing.pipeline import process_session
    from src.preprocessing.speaker import assign_speakers, interview_start

    bank = QuestionBank(cfg)
    src_dir = ROOT / "data/interim/transcripts"
    n_changed = 0
    for sid in sids:
        dst = interim_path(cfg, "transcripts", sid, "json")
        tr = json.load(open(src_dir / f"{sid}.json"))
        if reassign:
            old = [s["speaker"] for s in tr["segments"]]
            segs, info = assign_speakers(copy.deepcopy(tr["segments"]), load_wav(audio[sid], 16000), bank, cfg)
            t0 = interview_start(segs, bank, cfg.preprocessing.speaker.opening_min_score)
            n_changed += sum(o != s["speaker"] for o, s in zip(old, segs))
            tr.update(segments=[s for s in segs if s["end"] > t0 - 0.5], interview_start=t0, speaker=info)
        json.dump(tr, open(dst, "w"))
        vad = interim_path(cfg, "vad", sid, "json")
        if not vad.exists():
            vad.symlink_to(ROOT / f"data/interim/vad/{sid}.json")
        process_session(sid, audio[sid], cfg, ["qa"], "cpu", overwrite=True)
    n_pairs = [len(json.load(open(interim_path(cfg, "qa_pairs", s, "json")))["pairs"])
               if interim_path(cfg, "qa_pairs", s, "json").exists() else 0 for s in sids]
    return {"pairs_median": float(np.median(n_pairs)), "pairs_mean": float(np.mean(n_pairs)),
            "sessions_le3": int(sum(n <= 3 for n in n_pairs)), "segments_relabeled": int(n_changed)}


# ------------------------------------------------------------------ 3. judge
JUDGE = ("You are checking evidence for a PHQ-8 depression rating.\nItem: {item}\n"
         "Interview excerpt:\nQ: {q}\nA: {a}\n"
         "Does the participant's answer contain information that helps rate how often they experienced this item "
         "in the past two weeks? Answer yes or no.")


def judge(scorer, items, cands: dict) -> dict:
    """cands: key -> (k, question, answer). 返回 key -> P(yes)。"""
    import torch

    tok = scorer.tok
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    keys = list(cands)
    out = {}
    decoder, head = scorer._parts()
    for i in range(0, len(keys), 32):
        ks = keys[i:i + 32]
        prompts = [JUDGE.format(item=items[cands[k][0]]["text"], q=cands[k][1], a=" ".join(cands[k][2].split()[:150]))
                   for k in ks]
        enc = scorer.encode(prompts)
        with torch.inference_mode():
            h = decoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, -1]
            z = (h.to(head.weight.dtype) @ head.weight[yes + no].T).float()
        lp = torch.logsumexp(z[:, :2], 1) - torch.logsumexp(z, 1)
        for k, v in zip(ks, lp.exp().cpu().numpy()):
            out[k] = float(v)
    return out


def candidates(cfg, ret, sids, top_n):
    from src.preprocessing.pipeline import load_qa_pairs

    res = {}
    for sid in sids:
        pairs = {p["pair_id"]: p for p in load_qa_pairs(cfg, sid)}
        for k in range(1, N_ITEMS + 1):
            r = ret[sid][k]
            for rank, pid in enumerate(r["ranked"][:top_n]):
                p = pairs[pid]
                res[(sid, k, rank)] = (k, p["question"], p["answer"], r["scores"][rank], k in p["items"])
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--out", default=str(ROOT / "logs/stage1_bench.json"))
    ap.add_argument("--steps", nargs="*", default=["qa", "retrieve", "judge", "zeroshot", "lora"])
    ap.add_argument("--max-train", type=int, default=None, help="调试用：只用前 N 个 train session")
    a = ap.parse_args()
    work = Path(a.work) / "stage1_bench"
    work.mkdir(parents=True, exist_ok=True)
    out = json.load(open(a.out)) if Path(a.out).exists() else {}

    def save():
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    base = make_cfg(work, "old", str(ROOT / "resources/ellie_questions.csv"))
    lab = load_labels(base)
    audio = find_audio(base)
    train = [s for s in lab.loc[lab.split == "train", "sid"] if s in audio]
    dev = [s for s in lab.loc[lab.split == "dev", "sid"] if s in audio]
    if a.max_train:
        train = train[: a.max_train]
    allsids = train + dev
    log(f"train {len(train)} dev {len(dev)}")

    cfgs = {
        "R_old_bge": make_cfg(work, "old", str(ROOT / "resources/ellie_questions.csv")),
        "R_new_bge": make_cfg(work, "new", str(ROOT / "resources/ellie_questions_mined.csv")),
        "R_new_qwen": make_cfg(work, "new", str(ROOT / "resources/ellie_questions_mined.csv"), [
            f"stage1.retrieval.encoder={QWEN_EMB}"]),
        "ALL_new": make_cfg(work, "new", str(ROOT / "resources/ellie_questions_mined.csv"), [
            "stage1.retrieval.top_n=64", "stage1.retrieval.store_top=64", "stage1.retrieval.evidence_threshold=-1",
            "stage1.model.max_length=4096", "stage1.train.eval_batch_sessions=2"]),
    }

    # 不能走 --set：含 "Instruct: ...\nQuery:" 的字符串会被当成 YAML 解析成 dict
    cfgs["R_new_qwen"].stage1.retrieval.query_instruction = QWEN_INSTR

    if "qa" in a.steps:
        out["qa"] = {"old": build_qa(cfgs["R_old_bge"], allsids, audio, reassign=False)}
        log("qa old", out["qa"]["old"])
        out["qa"]["new"] = build_qa(cfgs["R_new_bge"], allsids, audio, reassign=True)
        log("qa new", out["qa"]["new"])
        save()

    from src.stage1.retrieval import build_retrieval, load_item_defs, load_retrieval, save_retrieval

    rets = {}
    for name, cfg in cfgs.items():
        p = work / f"ret_{name}.json"
        if "retrieve" in a.steps or not p.exists():
            save_retrieval(build_retrieval(cfg, allsids, "cuda"), p)
            import torch

            torch.cuda.empty_cache()
        rets[name] = load_retrieval(p)
        ev = [[rets[name][s][k]["ev"] for k in range(1, N_ITEMS + 1)] for s in dev]
        out.setdefault("retrieval", {})[name] = {"ev_rate_per_item": np.mean(ev, 0).round(3).tolist()}
    log("retrieval done")

    items = load_item_defs(base)
    if "judge" in a.steps:
        from src.stage1.model import ItemScorer

        scorer = ItemScorer(cfgs["R_old_bge"], "cuda", with_lora=False)
        scorer.model.eval()
        cands = {n: candidates(cfgs[n], rets[n], dev, 6) for n in ("R_old_bge", "R_new_bge", "R_new_qwen")}
        uniq = {}
        for c in cands.values():
            for k, q, ans, *_r in c.values():
                uniq[(k, q, ans)] = (k, q, ans)
        log(f"judging {len(uniq)} unique (item, Q-A) pairs")
        py = judge(scorer, items, uniq)
        for n, c in cands.items():
            rows = [(key[1], key[2], py[(v[0], v[1], v[2])], v[3], v[4]) for key, v in c.items()]
            arr = np.array([r[2] for r in rows])
            per_item = [float(np.mean([r[2] for r in rows if r[0] == k])) for k in range(1, N_ITEMS + 1)]
            top1 = [r[2] for r in rows if r[1] == 0]
            # 每个 (session, item) 至少有一个相关候选（P(yes) > 0.5）
            hit = {}
            for key, v in c.items():
                hit.setdefault((key[0], key[1]), []).append(py[(v[0], v[1], v[2])] > 0.5)
            out.setdefault("judge", {})[n] = {
                "mean_p_yes@6": float(arr.mean()), "precision@6": float((arr > 0.5).mean()),
                "precision@1": float(np.mean(np.array(top1) > 0.5)),
                "coverage_any_relevant@6": float(np.mean([any(v) for v in hit.values()])),
                "precision_per_item": [round(x, 3) for x in per_item],
                "precision_qmap_hits": float(np.mean([r[2] > 0.5 for r in rows if r[4]] or [np.nan])),
                "precision_embedding_hits": float(np.mean([r[2] > 0.5 for r in rows if not r[4]] or [np.nan]))}
            log("judge", n, out["judge"][n]["precision@6"])
        del scorer
        import torch

        torch.cuda.empty_cache()
        save()

    def evaluate(cfg, probs, sids):
        _, m = dzs.step_evaluate(cfg, lab, sids, probs)
        return {k: m[k] for k in ("total", "binary_phq10", "items", "pred_mean", "label_mean")}

    if "zeroshot" in a.steps:
        import torch

        from src.stage1.dataset import Stage1Sessions
        from src.stage1.model import ItemScorer
        from src.stage1.train import predict

        scorer = ItemScorer(cfgs["R_old_bge"], "cuda", with_lora=False)
        for name, cfg in cfgs.items():
            scorer.cfg = cfg                                  # max_length 等跟着配置走
            probs = predict(scorer, Stage1Sessions(cfg, dev, rets[name], None), cfg.stage1.train.eval_batch_sessions)
            out.setdefault("zeroshot", {})[name] = evaluate(cfg, probs, dev)
            t = out["zeroshot"][name]["total"]
            log("zeroshot", name, f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
            save()
        del scorer
        torch.cuda.empty_cache()

    if "lora" in a.steps:
        import torch

        from src.stage1.train import run_fold

        cand = [n for n in ("R_old_bge", "R_new_bge", "R_new_qwen") if n in out.get("judge", {})]
        best = max(cand, key=lambda n: out["judge"][n]["precision@6"]) if cand else "R_new_bge"
        runs = [best] + (["R_old_bge"] if best != "R_old_bge" else [])
        rng = random.Random(0)
        tr = list(train)
        rng.shuffle(tr)
        n_va = max(1, len(tr) // 5)
        va, tr = sorted(tr[:n_va]), sorted(tr[n_va:])
        for name in runs:
            cfg = cfgs[name]
            s1 = work / "outputs" / f"lora_{name}"
            s1.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            run_fold(cfg, 0, tr, va, {"dev": dev}, rets[name], lab, s1, "cuda")
            z = np.load(s1 / "fold0" / "probs.npz")
            hist = json.load(open(s1 / "fold0" / "history.json")) if (s1 / "fold0" / "history.json").exists() else {}
            out.setdefault("lora", {})[name] = {**evaluate(cfg, z["dev_probs"], z["dev_sids"].tolist()),
                                                "minutes": round((time.time() - t0) / 60, 1),
                                                "best_epoch": hist.get("best_epoch"), "n_train": len(tr),
                                                "n_val": len(va), "selected_by_judge": name == best}
            t = out["lora"][name]["total"]
            log("lora", name, f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
            save()
            torch.cuda.empty_cache()
    log("done ->", a.out)


if __name__ == "__main__":
    main()
