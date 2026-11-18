"""dev 集端到端测试：预处理 -> Stage 1 零样本（不训练 LoRA）-> 和 PHQ-8 标签比对。

只跑 data/raw/dev/ 下的 session（XXX_P/XXX_AUDIO.wav），标签来自 data/labels/dev_split.csv + Detailed_PHQ8_Labels.csv。
步骤（每步有缓存，任务中断后重新提交会从断点继续）：
  1. preprocess  vad -> transcribe -> qa（不做 segments / wavlm，Stage 1 用不到）
  2. retrieve    每题检索候选 Q-A 对（bge-large）
  3. score       LLM 读提示词，取 "0".."3" 四个 token 的概率 -> 每题期望分，8 题相加 = 总分
  4. evaluate    总分 CCC / RMSE / MAE / 分段 MAE，PHQ>=10 二分类 AUROC / F1，逐题 MAE / QWK

用法:
  python scripts/evaluation/dev_zeroshot.py --set stage1.model.base=Qwen/Qwen3-4B-Instruct-2507 \
      experiment.name=dev_zs_qwen3_4b_instruct
  python scripts/evaluation/dev_zeroshot.py --steps score evaluate --set ...   # 只换模型重打分
输出: outputs/<experiment.name>/dev_zeroshot/{retrieval.json, probs.npz, predictions.csv, metrics.json}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import N_ITEMS  # noqa: E402
from src.audio import find_audio  # noqa: E402
from src.config import base_argparser, config_from_args, exp_dir, resolve, save_config  # noqa: E402
from src.labels import ITEM_COLS, load_labels  # noqa: E402
from src.preprocessing import interim_path  # noqa: E402

STEPS = ["preprocess", "retrieve", "score", "evaluate"]


def dev_sids(cfg, lab) -> dict[int, Path]:
    audio = find_audio(cfg)
    dev_root = resolve(cfg.paths.raw_audio) / "dev"
    wanted = set(lab.loc[lab.split == "dev", "sid"])
    return {s: p for s, p in audio.items() if dev_root in p.parents and s in wanted}


def step_preprocess(cfg, sids: dict[int, Path], overwrite: bool) -> list[dict]:
    from src.preprocessing.pipeline import process_session

    log = []
    for i, (sid, wav) in enumerate(sorted(sids.items()), 1):
        t = time.time()
        try:
            st = process_session(sid, wav, cfg, ["vad", "transcribe", "qa"], cfg.experiment.device, overwrite)
        except Exception as e:                       # 单个 session 失败不影响其他
            traceback.print_exc()
            st = {"sid": sid, "error": repr(e)}
        st["sec"] = round(time.time() - t, 1)
        log.append(st)
        print(f"[preprocess {i}/{len(sids)}] {sid}: {'ok' if st.get('ok') else st.get('error')} ({st['sec']} s)",
              flush=True)
    return log


def step_score(cfg, sids: list[int], ret: dict, out: Path) -> np.ndarray:
    import torch

    from src.stage1.dataset import Stage1Sessions
    from src.stage1.model import ItemScorer

    scorer = ItemScorer(cfg, cfg.experiment.device, with_lora=False)
    scorer.model.eval()
    ds = Stage1Sessions(cfg, sids, ret, None)
    bs = cfg.stage1.train.eval_batch_sessions
    res = []
    with torch.inference_mode():                    # 不建计算图，否则冻结的 4B 参数也会存激活
        for i in range(0, len(ds), bs):
            batch = [ds.get(j) for j in range(i, min(i + bs, len(ds)))]
            z = scorer.score_logits([p for b in batch for p in b["prompts"]]).view(len(batch), N_ITEMS, -1)
            res.append(z.softmax(-1).cpu().numpy())
            print(f"[score] {min(i + bs, len(ds))}/{len(ds)}", flush=True)
    # 留一个例子方便检查提示词
    (out / "example_prompt.txt").write_text(scorer.format(ds.get(0)["prompts"][1]))
    return np.concatenate(res)


def step_evaluate(cfg, lab, sids: list[int], probs: np.ndarray) -> tuple[pd.DataFrame, dict]:
    from sklearn.metrics import f1_score

    from src.evaluation.metrics import item_report, safe_auc, total_report

    L = lab.set_index("sid").loc[sids]
    exp = (probs * np.arange(4)).sum(-1)                         # [N, 8] 每题期望分
    y_items = L[ITEM_COLS].to_numpy().astype(int)
    y = L["phq_total"].to_numpy().astype(float)
    yhat = exp.sum(1)
    pred = pd.DataFrame({"sid": sids, "phq_total": y, "y_text": yhat.round(3),
                         "y_binary": (y >= 10).astype(int), "pred_binary": (yhat >= 10).astype(int)})
    for k in range(N_ITEMS):
        pred[f"item{k + 1}"], pred[f"pred_item{k + 1}"] = y_items[:, k], exp[:, k].round(3)
    edges = cfg.evaluation.severity_bin_edges
    auroc, auprc = safe_auc((y >= 10).astype(int), yhat)
    m = {
        "n": len(sids), "model": cfg.stage1.model.base,
        "total": total_report(y, yhat, edges),
        "binary_phq10": {"auroc": auroc, "auprc": auprc,
                         "f1": float(f1_score(y >= 10, yhat >= 10, zero_division=0)),
                         "acc": float(((y >= 10) == (yhat >= 10)).mean()), "n_pos": int((y >= 10).sum())},
        "items": item_report(y_items, probs),
        "pred_mean": float(yhat.mean()), "label_mean": float(y.mean()),
        # 参考：总是预测 dev 标签均值（偏乐观，因为用了 dev 自己的均值）
        "const_dev_mean_rmse": float(np.sqrt(((y - y.mean()) ** 2).mean())),
        "items_sum_mismatch_sids": [int(s) for s, a, b in zip(sids, y_items.sum(1), y) if a != b],
    }
    return pred, m


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--sids", nargs="*", type=int, default=None, help="只跑 dev 里的这些 session")
    ap.add_argument("--overwrite", action="store_true", help="预处理缓存也重算")
    args = ap.parse_args()
    cfg = config_from_args(args)
    lab = load_labels(cfg)
    out = exp_dir(cfg, "dev_zeroshot")
    save_config(cfg, out)

    audio = dev_sids(cfg, lab)
    if args.sids:
        audio = {s: p for s, p in audio.items() if s in set(args.sids)}
    print(f"{len(audio)} dev sessions; model={cfg.stage1.model.base}; out={out}", flush=True)

    if "preprocess" in args.steps:
        log = step_preprocess(cfg, audio, args.overwrite)
        json.dump(log, open(out / "preprocess_log.json", "w"), indent=1)

    ok = sorted(s for s in audio if interim_path(cfg, "qa_pairs", s, "json").exists())
    if len(ok) < len(audio):
        print(f"!! {len(audio) - len(ok)} sessions 没有 Q-A 对，跳过: {sorted(set(audio) - set(ok))}")

    from src.stage1.retrieval import build_retrieval, load_retrieval, save_retrieval

    p_ret = out / "retrieval.json"
    if "retrieve" in args.steps:
        save_retrieval(build_retrieval(cfg, ok, cfg.experiment.device), p_ret)
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    ret = load_retrieval(p_ret)
    ok = [s for s in ok if s in ret]

    p_probs = out / "probs.npz"
    if "score" in args.steps:
        probs = step_score(cfg, ok, ret, out)
        np.savez(p_probs, sids=np.array(ok), probs=probs)
    if "evaluate" in args.steps:
        z = np.load(p_probs)
        pred, m = step_evaluate(cfg, lab, z["sids"].tolist(), z["probs"])
        pred.to_csv(out / "predictions.csv", index=False)
        json.dump(m, open(out / "metrics.json", "w"), indent=1, default=float)
        t, b = m["total"], m["binary_phq10"]
        print(f"\n== dev (n={m['n']}) {m['model']}")
        print(f"total  CCC {t['ccc']:.3f}  RMSE {t['rmse']:.2f}  MAE {t['mae']:.2f}  "
              f"(pred mean {m['pred_mean']:.1f} vs label {m['label_mean']:.1f}; const-mean RMSE "
              f"{m['const_dev_mean_rmse']:.2f})")
        print(f"PHQ>=10 AUROC {b['auroc']:.3f}  F1 {b['f1']:.3f}  acc {b['acc']:.3f}  (pos {b['n_pos']})")
        print("items  " + "  ".join(f"{k}: MAE {v['mae']:.2f} QWK {v['qwk']:.2f}" for k, v in m["items"].items()))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
