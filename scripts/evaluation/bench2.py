"""Benchmark v2：五折 × 多 seed 的 Stage 1、高分样本加权、严重程度区间任务、修正后的 Stage 2，bootstrap 置信区间。

输入配置（都用挖掘的问题库）
  R      ASR → Q-A 对 → Qwen3-Embedding 检索前 6 个
  ALLQA  ASR → Q-A 对，每题放全部 Q-A 对（max_length 4096）
  OFF    官方转写整份（max_length 5120，不经过音频）
训练变体
  base   原损失
  w      + 按总分的样本加权（stage1.loss.sample_weight=lds，w ∝ 密度^-0.5）
  bin    + 4 档严重程度区间任务（区间提示词 CE + 8 题卷积总分的区间一致性；推理时专家乘积）
  wbin   两者都加
OOF 修正：每个外层折内部再划 15% 做早停，外层验证折只用来出 OOF（不再既选 epoch 又打分）。

子命令
  run --group short|long --index i    跑任务表里第 i 个 (配置, 变体, 折, seed)，概率写到 logs/bench2/runs/
  tasks --group short|long            打印任务数
  aggregate                           汇总老师 → dev 指标 + bootstrap → Stage 2（修正版）+ β 融合 → logs/bench2/results.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import N_ITEMS  # noqa: E402
from src.audio import find_audio  # noqa: E402
from src.labels import ITEM_COLS, load_labels  # noqa: E402

_spec = importlib.util.spec_from_file_location("fb", ROOT / "scripts/evaluation/full_bench.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)
s1b = fb.s1b
log = s1b.log

RUNS = ROOT / "logs/bench2/runs"
N_FOLDS, SEEDS, INNER = 5, [0, 1], 0.15
VARIANTS = {"R": ["base", "w", "bin", "wbin"], "ALLQA": ["base", "bin", "wbin"], "OFF": ["base", "bin", "wbin"]}
GROUPS = {"short": ["R"], "long": ["OFF", "ALLQA"]}


def task_list(group):
    return [(c, v, f, s) for c in GROUPS[group] for v in VARIANTS[c] for s in SEEDS for f in range(N_FOLDS)]


def make_cfg(work, config, variant, seed=0):
    common = [f"paths.features={work / 'features'}", f"paths.splits={work / 'splits'}",
              f"experiment.seed={42 + 100 * seed}"]
    if "w" in variant.replace("bin", ""):
        common.append("stage1.loss.sample_weight=lds")
    if "bin" in variant:
        common.append("stage1.severity.enabled=true")
    if config == "R":
        # 4096 只影响区间提示词（8 道题的候选并集）；逐题提示词单独成 batch，长度不变
        cfg = s1b.make_cfg(work, "new", fb.BANK, [f"stage1.retrieval.encoder={s1b.QWEN_EMB}",
                                                  "stage1.model.max_length=4096", *common])
        cfg.stage1.retrieval.query_instruction = s1b.QWEN_INSTR
    elif config == "ALLQA":
        cfg = s1b.make_cfg(work, "new", fb.BANK, [
            "stage1.retrieval.top_n=64", "stage1.retrieval.store_top=64", "stage1.retrieval.evidence_threshold=-1",
            "stage1.augment.n_min=64", "stage1.augment.n_max=64", "stage1.augment.drop_one_prob=0",
            "stage1.augment.shuffle_prob=0", "stage1.model.max_length=4096",
            "stage1.train.eval_batch_sessions=2", *common])
    else:
        cfg = s1b.make_cfg(work, "new", fb.BANK, ["stage1.model.max_length=5120",
                                                  "stage1.train.eval_batch_sessions=1", *common])
    return cfg


def sids_and_folds(cfg):
    from src.splits import make_folds

    lab = load_labels(cfg)
    audio = find_audio(cfg)
    train = [s for s in lab.loc[lab.split == "train", "sid"] if s in audio]
    dev = [s for s in lab.loc[lab.split == "dev", "sid"] if s in audio]
    folds = make_folds(lab, cfg)
    folds = folds[folds.sid.isin(train)].reset_index(drop=True)
    return lab, audio, train, dev, folds


# ------------------------------------------------------------------ run
def cmd_run(a):
    config, variant, f, seed = task_list(a.group)[a.index]
    tag = f"{config}_{variant}_f{f}_s{seed}"
    dst = RUNS / f"{tag}.npz"
    if dst.exists():
        log("exists", dst)
        return
    work = Path(a.work) / f"bench2_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(work, config, variant, seed)
    lab, audio, train, dev, folds = sids_and_folds(cfg)
    log(f"task {tag}: train {len(train)} dev {len(dev)}")

    import src.stage1.train as s1t

    if config == "OFF":
        s1t.Stage1Sessions = fb.OfficialSessions
        ret = None
    else:
        from src.stage1.retrieval import build_retrieval
        import torch

        s1b.build_qa(cfg, train + dev, audio, reassign=True)
        ret = build_retrieval(cfg, train + dev, "cuda")
        torch.cuda.empty_cache()

    tr = folds.loc[folds.fold != f, "sid"].tolist()
    va = folds.loc[folds.fold == f, "sid"].tolist()
    rng = np.random.default_rng(1000 * seed + f)
    perm = rng.permutation(tr).tolist()
    n_in = int(round(len(perm) * INNER))
    stop, fit = sorted(perm[:n_in]), sorted(perm[n_in:])
    t0 = time.time()
    s1dir = work / "stage1"
    s1t.run_fold(cfg, f, fit, stop, {"oof": va, "dev": dev}, ret, lab, s1dir, "cuda")
    z = dict(np.load(s1dir / f"fold{f}" / "probs.npz"))
    hist = json.load(open(s1dir / f"fold{f}" / "history.json"))
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **{k: v for k, v in z.items() if not k.startswith("val_")},
             best_epoch=hist["best_epoch"], minutes=(time.time() - t0) / 60)
    y = lab.set_index("sid").loc[z["dev_sids"], "phq_total"].to_numpy()
    yhat = s1t.predict_total(cfg, z["dev_probs"], z.get("dev_bin"))
    from src.evaluation.metrics import ccc

    log(f"{tag}: best_epoch {hist['best_epoch']}  dev CCC {ccc(y, yhat):.3f}  MAE {np.abs(y - yhat).mean():.2f}"
        f"  {(time.time() - t0) / 60:.1f} min -> {dst}")


# ------------------------------------------------------------------ aggregate
def metrics(lab, sids, yhat, probs=None, edges=(0, 5, 10, 15, 20, 25)):
    from sklearn.metrics import f1_score

    from src.evaluation.metrics import item_report, safe_auc, total_report

    L = lab.set_index("sid").loc[list(sids)]
    y = L.phq_total.to_numpy().astype(float)
    auroc, auprc = safe_auc((y >= 10).astype(int), yhat)
    m = {"total": total_report(y, yhat, list(edges)), "pred_mean": float(np.mean(yhat)),
         "binary_phq10": {"auroc": auroc, "auprc": auprc, "f1": float(f1_score(y >= 10, yhat >= 10)),
                          "recall": float(((yhat >= 10) & (y >= 10)).sum() / max(1, (y >= 10).sum()))}}
    if probs is not None:
        m["items"] = item_report(L[ITEM_COLS].to_numpy().astype(int), probs)
    return m


def bootstrap(y, preds: dict, ref: str | None, n=2000, seed=0):
    """每个预测的 CCC / MAE 的 95% CI；与 ref 的配对差值 CI 和 P(差值 <= 0)。"""
    from src.evaluation.metrics import ccc

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    out = {}
    stats = {k: np.array([[ccc(y[i], p[i]), np.abs(y[i] - p[i]).mean()] for i in idx]) for k, p in preds.items()}
    for k, st in stats.items():
        out[k] = {"ccc_ci": np.percentile(st[:, 0], [2.5, 97.5]).tolist(),
                  "mae_ci": np.percentile(st[:, 1], [2.5, 97.5]).tolist()}
        if ref and k != ref:
            d = st[:, 0] - stats[ref][:, 0]
            out[k][f"dccc_vs_{ref}"] = {"mean": float(d.mean()), "ci": np.percentile(d, [2.5, 97.5]).tolist(),
                                        "p_le_0": float((d <= 0).mean())}
    return out


def cmd_aggregate(a):
    import src.stage1.train as s1t
    from src.stage1.teacher import save_teacher

    work = Path(a.work) / "bench2_agg"
    work.mkdir(parents=True, exist_ok=True)
    cfg0 = make_cfg(work, "R", "base")
    lab, audio, train, dev, folds = sids_and_folds(cfg0)
    labi = lab.set_index("sid")
    y_dev = labi.loc[dev, "phq_total"].to_numpy().astype(float)
    res = {"setup": {"n_folds": N_FOLDS, "seeds": SEEDS, "inner_stop_frac": INNER, "n_train": len(train),
                     "n_dev": len(dev)}, "stage1": {}}
    teachers, dev_preds = {}, {}

    def save():
        json.dump(res, open(ROOT / "logs/bench2/results.json", "w"), indent=1, default=float)

    for config, vs in VARIANTS.items():
        for v in vs:
            key = f"{config}_{v}"
            files = {(f, s): RUNS / f"{key}_f{f}_s{s}.npz" for f in range(N_FOLDS) for s in SEEDS}
            if not all(p.exists() for p in files.values()):
                res["stage1"][key] = {"missing": [p.name for p in files.values() if not p.exists()]}
                continue
            cfg = make_cfg(work, config, v)
            Z = {k: dict(np.load(p)) for k, p in files.items()}
            has_bin = "dev_bin" in Z[(0, 0)]
            # dev：10 个模型平均；每个 seed 单独的 5 折集成也算一下（看 seed 方差）
            pd_ = np.mean([z["dev_probs"] for z in Z.values()], 0)
            pb = np.mean([z["dev_bin"] for z in Z.values()], 0) if has_bin else None
            yd = s1t.predict_total(cfg, pd_, pb)
            per_seed = []
            for s in SEEDS:
                ps = np.mean([Z[(f, s)]["dev_probs"] for f in range(N_FOLDS)], 0)
                bs = np.mean([Z[(f, s)]["dev_bin"] for f in range(N_FOLDS)], 0) if has_bin else None
                per_seed.append(metrics(lab, dev, s1t.predict_total(cfg, ps, bs))["total"]["ccc"])
            # train OOF：每个 session 在它的外层验证折上，seed 平均
            oof_s, oof_p, oof_b = [], [], []
            for f in range(N_FOLDS):
                zs = [Z[(f, s)] for s in SEEDS]
                oof_s += zs[0]["oof_sids"].tolist()
                oof_p.append(np.mean([z["oof_probs"] for z in zs], 0))
                if has_bin:
                    oof_b.append(np.mean([z["oof_bin"] for z in zs], 0))
            oof_p = np.concatenate(oof_p)
            oof_b = np.concatenate(oof_b) if has_bin else None
            yo = s1t.predict_total(cfg, oof_p, oof_b)
            r = {"dev": metrics(lab, dev, yd, pd_), "dev_ccc_per_seed": per_seed,
                 "train_oof": metrics(lab, oof_s, yo)["total"],
                 "best_epochs": [int(z["best_epoch"]) for z in Z.values()],
                 "minutes_per_run": float(np.mean([z["minutes"] for z in Z.values()]))}
            if has_bin:   # 不做专家乘积、只用 8 题之和的结果，以及区间题自己的准确率
                from src.stage1.severity import bin_index

                r["dev_items_sum_only"] = metrics(lab, dev, s1t.predict_total(cfg, pd_, None))["total"]
                r["dev_bin_head_acc"] = float((pb.argmax(1) == bin_index(y_dev)).mean())
                dev_preds[key + "_sumonly"] = s1t.predict_total(cfg, pd_, None)
            res["stage1"][key] = r
            dev_preds[key] = yd
            tp = work / f"teacher_{key}.npz"
            save_teacher(tp, oof_s + list(dev), ["train"] * len(oof_s) + ["dev"] * len(dev),
                         np.concatenate([oof_p, pd_]), y_text=np.concatenate([yo, yd]))
            teachers[key] = tp
            t = r["dev"]["total"]
            log(key, f"dev CCC {t['ccc']:.3f} RMSE {t['rmse']:.2f} MAE {t['mae']:.2f} per-seed {np.round(per_seed, 3)}")
    res["bootstrap_stage1"] = bootstrap(y_dev, dev_preds, "R_base" if "R_base" in dev_preds else None)
    for c in VARIANTS:                           # 各配置内部：每个变体相对 base，wbin 相对 bin
        for k, ref in [(f"{c}_{v}", f"{c}_base") for v in VARIANTS[c][1:]] + [(f"{c}_wbin", f"{c}_bin")]:
            if ref in dev_preds and k in dev_preds and ref != "R_base":
                res["bootstrap_stage1"][k][f"dccc_vs_{ref}"] = bootstrap(
                    y_dev, {ref: dev_preds[ref], k: dev_preds[k]}, ref)[k][f"dccc_vs_{ref}"]
    save()

    # ---------------- Stage 2（修正版）
    if a.skip_stage2 or not teachers:
        return
    import pandas as pd
    import torch

    from src.evaluation.metrics import ccc
    from src.fusion import fuse
    from src.preprocessing.pipeline import process_session
    from src.stage2.dataset import Bags
    from src.stage2.train import predictions_frame, run_cv

    s1b.build_qa(cfg0, train + dev, audio, reassign=True)
    for sid in train + dev:
        process_session(sid, audio[sid], cfg0, ["segments", "wavlm"], "cuda", overwrite=True)
    torch.cuda.empty_cache()
    log("wavlm features ready")
    grid = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
    res["stage2"] = {}
    s2_runs = [(k, 0.2) for k in teachers] + ([("R_base", None)] if "R_base" in teachers else [])
    for key, inner in s2_runs:
        name = key + ("" if inner else "__old_early_stop")
        cfg0.stage2.train.inner_val_frac = inner
        bags = Bags(cfg0, folds.sid.tolist() + dev, teachers[key], lab)
        si = {s: i for i, s in enumerate(bags.sids)}
        t0 = time.time()
        rr = run_cv(cfg0, bags, folds, {"dev": [si[s] for s in dev]}, "cuda")
        pr = predictions_frame(bags, rr["train_idx"], rr["oof"], "train").merge(folds[["sid", "fold"]], on="sid")
        pdv = predictions_frame(bags, [si[s] for s in dev], rr["eval"]["dev"], "dev")
        yt = labi.loc[pr.sid, "phq_total"].to_numpy()
        scores = {b: ccc(yt, fuse(pr.y_text.to_numpy(), pr.y_tot.to_numpy(), b)) for b in grid}
        beta = max(scores, key=scores.get)
        # 嵌套：每折用其余 4 折的 OOF 选 β，得到诚实的 train 融合 CCC
        nested = np.zeros(len(pr))
        betas = []
        for f in range(N_FOLDS):
            m = (pr.fold == f).to_numpy()
            sc = {b: ccc(yt[~m], fuse(pr.y_text[~m].to_numpy(), pr.y_tot[~m].to_numpy(), b)) for b in grid}
            bf = max(sc, key=sc.get)
            betas.append(bf)
            nested[m] = fuse(pr.y_text[m].to_numpy(), pr.y_tot[m].to_numpy(), bf)
        yf = fuse(pdv.y_text.to_numpy(), pdv.y_tot.to_numpy(), beta)
        res["stage2"][name] = {
            "beta": beta, "beta_scores_oof": scores, "nested_betas": betas,
            "train_oof_ccc": {"text": ccc(yt, pr.y_text), "audio_tot": ccc(yt, pr.y_tot), "fused_beta": scores[beta],
                              "fused_nested": ccc(yt, nested)},
            "dev": {"y_text": metrics(lab, dev, pdv.y_text.to_numpy())["total"],
                    "y_tot": metrics(lab, dev, pdv.y_tot.to_numpy())["total"],
                    "y_final": metrics(lab, dev, yf)},
            "bootstrap_final_vs_text": bootstrap(y_dev, {"text": pdv.y_text.to_numpy(), "final": yf}, "text")["final"],
            "best_epochs": [r["best_epoch"] for r in rr["log"]], "minutes": (time.time() - t0) / 60}
        d = res["stage2"][name]["dev"]
        log("stage2", name, "beta", beta, "dev CCC text", round(d["y_text"]["ccc"], 3), "tot",
            round(d["y_tot"]["ccc"], 3), "final", round(d["y_final"]["total"]["ccc"], 3))
        save()
    log("done -> logs/bench2/results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tasks", "aggregate"])
    ap.add_argument("--group", choices=list(GROUPS), default="short")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--skip-stage2", action="store_true")
    a = ap.parse_args()
    if a.cmd == "tasks":
        for i, t in enumerate(task_list(a.group)):
            print(i, *t)
    elif a.cmd == "run":
        if a.index is None:
            a.index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        cmd_run(a)
    else:
        cmd_aggregate(a)


if __name__ == "__main__":
    main()
