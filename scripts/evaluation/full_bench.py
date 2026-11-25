"""完整管线 dev benchmark（挖掘的问题库 + Qwen3-Embedding），以及不检索的对照。

  --part full    preprocessing(qa → segments → WavLM) → Stage 1 五折 LoRA（OOF 老师）→ Stage 2 音频残差（5 折 × 5 seed）
                 → β 融合（在 train OOF 上选）→ dev 评估
  --part allqa   不做 embedding 筛选：每道题都把整场访谈的全部 Q-A 对（按时间）放进提示词，仍然输出 8 道题的分数；
                 Stage 1 LoRA 训练一次（train 80/20 早停，与 stage1_bench 的 LoRA 相同划分），dev 评估
  --part official  不用音频：直接把官方 XXX_Transcript.csv 整份（按时间、无说话人标签）放进每道题的提示词，
                 max_length 5120（最长 session ~4.7k token，不截断）；Stage 1 训练一次，划分同上，dev 评估

中间产物（Q-A、特征、LoRA 权重）全写在 --work（默认 $TMPDIR，作业结束自动删除），只留 --out 一个 JSON。
  python scripts/evaluation/full_bench.py --part full  --out logs/full_bench_full.json
  python scripts/evaluation/full_bench.py --part allqa --out logs/full_bench_allqa.json
"""
import argparse
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
from src.audio import find_audio  # noqa: E402
from src.labels import load_labels  # noqa: E402

_spec = importlib.util.spec_from_file_location("s1b", ROOT / "scripts/evaluation/stage1_bench.py")
s1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1b)
log = s1b.log
BANK = str(ROOT / "resources/ellie_questions_mined.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["full", "allqa", "official"], required=True)
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-train", type=int, default=None, help="调试用")
    a = ap.parse_args()
    a.out = a.out or str(ROOT / f"logs/full_bench_{a.part}.json")
    work = Path(a.work) / f"full_bench_{a.part}"
    work.mkdir(parents=True, exist_ok=True)
    out = {}

    def save():
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    common = [f"paths.features={work / 'features'}", f"paths.splits={work / 'splits'}"]
    if a.part == "full":
        cfg = s1b.make_cfg(work, "new", BANK, [f"stage1.retrieval.encoder={s1b.QWEN_EMB}", *common])
        cfg.stage1.retrieval.query_instruction = s1b.QWEN_INSTR
    elif a.part == "official":
        cfg = s1b.make_cfg(work, "new", BANK, ["stage1.model.max_length=5120", "stage1.train.eval_batch_sessions=1",
                                               *common])
    else:
        cfg = s1b.make_cfg(work, "new", BANK, [
            "stage1.retrieval.top_n=64", "stage1.retrieval.store_top=64", "stage1.retrieval.evidence_threshold=-1",
            "stage1.augment.n_min=64", "stage1.augment.n_max=64", "stage1.augment.drop_one_prob=0",
            "stage1.augment.shuffle_prob=0", "stage1.model.max_length=4096",
            "stage1.train.eval_batch_sessions=2", *common])
    lab = load_labels(cfg)
    audio = find_audio(cfg)
    train = [s for s in lab.loc[lab.split == "train", "sid"] if s in audio]
    dev = [s for s in lab.loc[lab.split == "dev", "sid"] if s in audio]
    if a.max_train:
        train = train[: a.max_train]
    allsids = train + dev
    log(f"part {a.part}: train {len(train)} dev {len(dev)}")

    if a.part == "official":
        return run_official(cfg, lab, train, dev, work, out, save, a.out)

    # ---------------- preprocessing
    out["qa"] = s1b.build_qa(cfg, allsids, audio, reassign=True)
    log("qa", out["qa"])
    save()
    if a.part == "full":
        from src.preprocessing.pipeline import process_session

        t0 = time.time()
        for sid in allsids:
            process_session(sid, audio[sid], cfg, ["segments", "wavlm"], "cuda", overwrite=True)
        import pandas as pd
        import torch

        from src.preprocessing import interim_path

        nseg = [len(pd.read_csv(interim_path(cfg, "segments", s, "csv"))) for s in allsids]
        out["segments"] = {"median": float(np.median(nseg)), "mean": float(np.mean(nseg)),
                           "minutes": round((time.time() - t0) / 60, 1)}
        log("segments + wavlm", out["segments"])
        torch.cuda.empty_cache()

    # ---------------- retrieval
    import torch

    from src.stage1.retrieval import build_retrieval

    ret = build_retrieval(cfg, allsids, "cuda")
    torch.cuda.empty_cache()
    log("retrieval done")

    def evaluate_probs(probs, sids):
        _, m = s1b.dzs.step_evaluate(cfg, lab, sids, probs)
        return {k: m[k] for k in ("total", "binary_phq10", "items", "pred_mean", "label_mean")}

    from src.stage1.train import run_fold

    if a.part == "allqa":
        rng = random.Random(0)                       # 与 stage1_bench 相同的 80/20 划分
        tr = list(train)
        rng.shuffle(tr)
        n_va = max(1, len(tr) // 5)
        va, tr = sorted(tr[:n_va]), sorted(tr[n_va:])
        s1 = work / "outputs" / "lora_allqa"
        s1.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run_fold(cfg, 0, tr, va, {"dev": dev}, ret, lab, s1, "cuda")
        z = np.load(s1 / "fold0" / "probs.npz")
        hist = json.load(open(s1 / "fold0" / "history.json"))
        out["stage1_lora"] = {**evaluate_probs(z["dev_probs"], z["dev_sids"].tolist()),
                              "minutes": round((time.time() - t0) / 60, 1), "best_epoch": hist.get("best_epoch"),
                              "n_train": len(tr), "n_val": len(va)}
        t = out["stage1_lora"]["total"]
        log("allqa LoRA", f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
        save()
        log("done ->", a.out)
        return

    # ---------------- Stage 1：五折 OOF 老师
    from src.splits import fold_ids, make_folds
    from src.stage1.teacher import load_teacher
    from src.stage1.train import assemble_oof

    folds = make_folds(lab, cfg)
    folds = folds[folds.sid.isin(train)].reset_index(drop=True)
    (work / "splits").mkdir(parents=True, exist_ok=True)
    folds.to_csv(work / "splits" / "folds.csv", index=False)
    n_folds = int(cfg.splits.n_folds)
    s1 = work / "outputs" / "stage1"
    s1.mkdir(parents=True, exist_ok=True)
    out["stage1_folds"] = []
    for f in range(n_folds):
        tr, va = fold_ids(folds, f)
        t0 = time.time()
        run_fold(cfg, f, tr, va, {"dev": dev}, ret, lab, s1, "cuda")
        z = np.load(s1 / f"fold{f}" / "probs.npz")
        hist = json.load(open(s1 / f"fold{f}" / "history.json"))
        m = evaluate_probs(z["dev_probs"], z["dev_sids"].tolist())["total"]
        out["stage1_folds"].append({"fold": f, "best_epoch": hist.get("best_epoch"), "dev_ccc": m["ccc"],
                                    "dev_rmse": m["rmse"], "dev_mae": m["mae"],
                                    "minutes": round((time.time() - t0) / 60, 1)})
        log(f"stage1 fold {f}", out["stage1_folds"][-1])
        save()
    tpath = assemble_oof(s1, n_folds, ["dev"])
    teacher = load_teacher(tpath)
    tm = teacher["split"] == "dev"
    out["stage1_ensemble_dev"] = evaluate_probs(teacher["probs"][tm], teacher["sids"][tm].tolist())
    trm = teacher["split"] == "train"
    out["stage1_oof_train"] = evaluate_probs(teacher["probs"][trm], teacher["sids"][trm].tolist())["total"]
    t = out["stage1_ensemble_dev"]["total"]
    log("stage1 5-fold ensemble dev", f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
    save()

    # ---------------- Stage 2：音频残差
    import pandas as pd

    from src.evaluation.report import build_final, evaluate
    from src.stage2.dataset import Bags
    from src.stage2.train import predictions_frame, run_cv

    bags = Bags(cfg, folds.sid.tolist() + dev, tpath, lab)
    sid_to_i = {s: i for i, s in enumerate(bags.sids)}
    eval_idx = {"dev": [sid_to_i[s] for s in dev]}
    t0 = time.time()
    s2 = work / "outputs" / "stage2"
    s2.mkdir(parents=True, exist_ok=True)
    res = run_cv(cfg, bags, folds, eval_idx, "cuda", save_dir=None)
    pred = pd.concat([predictions_frame(bags, res["train_idx"], res["oof"], "train"),
                      predictions_frame(bags, eval_idx["dev"], res["eval"]["dev"], "dev")])
    final, finfo = build_final(pred, lab, list(cfg.fusion.beta_grid))
    met = evaluate(final, lab, teacher, cfg)
    out["stage2"] = {"cv_ccc_train_oof": res["cv_ccc"], "minutes": round((time.time() - t0) / 60, 1),
                     "best_epochs": [r["best_epoch"] for r in res["log"]], "fusion": finfo,
                     "dev": met.get("dev"), "train_oof": met.get("train")}
    d = met["dev"]
    log("stage2 dev", {c: round(d[c]["ccc"], 3) for c in ("y_text", "y_tot", "y_final")}, "beta", finfo["beta"])
    save()
    log("done ->", a.out)


class OfficialSessions:
    """和 Stage1Sessions 接口相同；每道题的证据 = 整份官方转写（每行一句，按时间）。"""

    def __init__(self, cfg, sids, retrieval=None, labels=None, train=False, seed=0):
        import pandas as pd

        from src.stage1.retrieval import load_item_defs

        self.cfg, self.sids = cfg, list(sids)
        self.items = load_item_defs(cfg)
        self.text = {}
        for sid in self.sids:
            f = next(iter((ROOT / "data/raw").glob(f"*/{sid}_P/{sid}_Transcript.csv")))
            d = pd.read_csv(f).sort_values("Start_Time")
            self.text[sid] = "\n".join(t.strip() for t in d.Text.dropna().astype(str) if t.strip())
        self.labels = labels.set_index("sid") if labels is not None else None

    def __len__(self):
        return len(self.sids)

    def get(self, idx):
        from src import N_ITEMS

        sid = self.sids[idx]
        prompts = ["\n".join([
            "You are rating one item of the PHQ-8 for the past two weeks.",
            f"Item {k}: {self.items[k]['text']}. 0 = not at all, 1 = several days, "
            "2 = more than half the days, 3 = nearly every day.",
            "Full interview transcript (interviewer and participant, no speaker labels, in time order):",
            self.text[sid], "Answer with a single digit 0-3.", "Score:"]) for k in range(1, N_ITEMS + 1)]
        out = {"sid": sid, "prompts": prompts}
        from src.stage1.dataset import bin_enabled

        if bin_enabled(self.cfg):
            from src.stage1.prompts import build_bin_prompt

            out["bin_prompt"] = build_bin_prompt(["Full interview transcript (interviewer and participant, "
                                                  "no speaker labels):", self.text[sid]])
        if self.labels is not None:
            row = self.labels.loc[sid]
            out["y_items"] = np.array([row[f"item{k}"] for k in range(1, N_ITEMS + 1)], dtype=np.int64)
            out["y_total"] = float(row["phq_total"])
        return out


def run_official(cfg, lab, train, dev, work, out, save, out_path):
    import src.stage1.train as s1t
    from src.stage1.model import ItemScorer

    s1t.Stage1Sessions = OfficialSessions            # run_fold 内部构造数据集时用官方转写
    ds = OfficialSessions(cfg, dev)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.stage1.model.base)
    lens = [len(tok(OfficialSessions(cfg, [s]).get(0)["prompts"][0])["input_ids"]) for s in train + dev]
    out["prompt_tokens"] = {"median": float(np.median(lens)), "max": int(max(lens)),
                            "n_over_max_length": int(sum(n > cfg.stage1.model.max_length for n in lens))}
    log("prompt tokens", out["prompt_tokens"])

    def evaluate_probs(probs, sids):
        _, m = s1b.dzs.step_evaluate(cfg, lab, sids, probs)
        return {k: m[k] for k in ("total", "binary_phq10", "items", "pred_mean", "label_mean")}

    # 零样本（无 LoRA）作参考
    import torch

    scorer = ItemScorer(cfg, "cuda", with_lora=False)
    out["zeroshot"] = evaluate_probs(s1t.predict(scorer, ds, 1), dev)
    t = out["zeroshot"]["total"]
    log("official zeroshot", f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
    del scorer
    torch.cuda.empty_cache()
    save()

    rng = random.Random(0)                           # 与 stage1_bench 相同的 80/20 划分
    tr = list(train)
    rng.shuffle(tr)
    n_va = max(1, len(tr) // 5)
    va, tr = sorted(tr[:n_va]), sorted(tr[n_va:])
    s1 = work / "outputs" / "lora_official"
    s1.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s1t.run_fold(cfg, 0, tr, va, {"dev": dev}, None, lab, s1, "cuda")
    z = np.load(s1 / "fold0" / "probs.npz")
    hist = json.load(open(s1 / "fold0" / "history.json"))
    out["stage1_lora"] = {**evaluate_probs(z["dev_probs"], z["dev_sids"].tolist()),
                          "minutes": round((time.time() - t0) / 60, 1), "best_epoch": hist.get("best_epoch"),
                          "n_train": len(tr), "n_val": len(va)}
    t = out["stage1_lora"]["total"]
    log("official LoRA", f"CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f}")
    save()
    log("done ->", out_path)


if __name__ == "__main__":
    main()
