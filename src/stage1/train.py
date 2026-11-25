"""Stage 1 training: per-fold LoRA fine-tuning, early stopping on val MAE, OOF teacher generation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .. import N_ITEMS
from ..config import Cfg, set_seed
from .dataset import Stage1Sessions, bin_enabled
from .losses import class_weights, session_weights, stage1_loss
from .model import ItemScorer
from .teacher import save_teacher


def _sev(cfg: Cfg):
    return cfg.stage1.get("severity", {})


@torch.no_grad()
def predict(scorer: ItemScorer, ds: Stage1Sessions, batch_sessions: int, return_bin: bool = False):
    """[len(ds), 8, 4] probabilities；return_bin=True 时另外返回区间题的分布 [len(ds), 4]（没开区间题则为 None）。"""
    scorer.model.eval()
    out, outb = [], []
    use_bin = bin_enabled(scorer.cfg)
    for i in range(0, len(ds), batch_sessions):
        batch = [ds.get(j) for j in range(i, min(i + batch_sessions, len(ds)))]
        prompts = [p for b in batch for p in b["prompts"]]
        z = scorer.score_logits(prompts).view(len(batch), N_ITEMS, -1)
        out.append(z.softmax(-1).cpu().numpy())
        if use_bin:
            outb.append(scorer.score_logits([b["bin_prompt"] for b in batch]).softmax(-1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, N_ITEMS, 4), dtype=np.float32)
    if not return_bin:
        return probs
    return probs, (np.concatenate(outb) if outb else None)


def predict_total(cfg: Cfg, probs: np.ndarray, bin_probs: np.ndarray | None) -> np.ndarray:
    from .severity import DEFAULT_EDGES, combine

    sv = _sev(cfg)
    return combine(probs, bin_probs if sv.get("combine", True) else None, sv.get("edges", DEFAULT_EDGES),
                   sv.get("gamma", 1.0))


def _val_mae(scorer: ItemScorer, ds: Stage1Sessions, batch_sessions: int) -> float:
    probs, pb = predict(scorer, ds, batch_sessions, return_bin=True)
    y = np.array([ds.labels.loc[s, "phq_total"] for s in ds.sids])
    return float(np.abs(predict_total(scorer.cfg, probs, pb) - y).mean())


def train_model(cfg: Cfg, scorer: ItemScorer, train_ds: Stage1Sessions, val_ds: Stage1Sessions | None,
                out_dir: Path, fixed_epochs: int | None = None) -> dict:
    """Train LoRA in place; restores the best (by val MAE) LoRA weights at the end."""
    tc, lc = cfg.stage1.train, cfg.stage1.loss
    y_items = np.stack([train_ds.get(i)["y_items"] for i in range(len(train_ds))])
    w = class_weights(y_items, lc.class_weight, lc.class_weight_smoothing)
    y_tot_tr = np.array([train_ds.get(i)["y_total"] for i in range(len(train_ds))])
    swmap = session_weights(y_tot_tr, lc.get("sample_weight", "none"), lc.get("sample_weight_sigma", 2.0),
                            lc.get("sample_weight_power", 0.5))
    sv = _sev(cfg)
    use_bin = bin_enabled(cfg)
    lam_cons, lam_head = float(sv.get("lambda_cons", 0.0)), float(sv.get("lambda_head", 0.0))
    edges = list(sv.get("edges", [0, 5, 10, 15, 25]))
    if use_bin:
        from .severity import bin_index

    params = [p for p in scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / tc.sessions_per_step)
    n_epochs = fixed_epochs or tc.max_epochs
    total = steps_per_epoch * n_epochs
    warm = max(1, int(tc.warmup_ratio * total))
    def lr_lambda(s: int) -> float:   # 线性预热，然后线性衰减到 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = {"mae": float("inf"), "epoch": -1, "state": None}
    history, bad = [], 0
    order = list(range(len(train_ds)))
    for epoch in range(n_epochs):
        scorer.model.train()
        random.shuffle(order)
        losses = []
        for s0 in range(0, len(order), tc.sessions_per_step):
            step_idx = order[s0: s0 + tc.sessions_per_step]
            opt.zero_grad(set_to_none=True)
            for m0 in range(0, len(step_idx), tc.micro_batch_sessions):   # 梯度累积
                batch = [train_ds.get(i) for i in step_idx[m0: m0 + tc.micro_batch_sessions]]
                prompts = [p for b in batch for p in b["prompts"]]
                z = scorer.score_logits(prompts).view(len(batch), N_ITEMS, -1)
                yi = torch.tensor(np.stack([b["y_items"] for b in batch]), device=z.device)
                yt = torch.tensor([b["y_total"] for b in batch], device=z.device, dtype=torch.float32)
                sw = torch.tensor([swmap[int(b["y_total"])] for b in batch], device=z.device)
                loss, parts = stage1_loss(z, yi, yt, w, lc.mu, lc.lambda_sum, sess_w=sw,
                                          lambda_bin=lam_cons if use_bin else 0.0, bin_edges=edges)
                (loss * len(batch) / len(step_idx)).backward()
                losses.append(loss.item())
                if use_bin and lam_head > 0:     # 区间题单独前向 / 反向，显存峰值不增加
                    zb = scorer.score_logits([b["bin_prompt"] for b in batch])
                    tb = torch.tensor(bin_index(yt.cpu().numpy(), edges), device=zb.device)
                    lb = torch.nn.functional.cross_entropy(zb, tb, reduction="none")
                    lb = (sw * lb).mean()
                    (lam_head * lb * len(batch) / len(step_idx)).backward()
            torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm)
            opt.step()
            sched.step()

        rec = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_ds is not None:
            rec["val_mae"] = _val_mae(scorer, val_ds, tc.eval_batch_sessions)
            if rec["val_mae"] < best["mae"]:
                best.update(mae=rec["val_mae"], epoch=epoch, state=scorer.lora_state())
                bad = 0
            else:
                bad += 1
        history.append(rec)
        print(json.dumps(rec))
        if val_ds is not None and fixed_epochs is None and bad >= tc.patience:
            break

    if best["state"] is not None:
        scorer.load_lora_state(best["state"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer.model.save_pretrained(out_dir / "adapter")
    summary = {"best_epoch": best["epoch"], "best_val_mae": best["mae"], "history": history}
    with open(out_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_fold(cfg: Cfg, f: int, train_sids, val_sids, eval_sids: dict[str, list[int]], retrieval, labels,
             stage1_dir: Path, device: str) -> None:
    """Train fold model M_f, then save its probabilities for val (OOF) and dev/test sessions."""
    set_seed(cfg.experiment.seed + f)
    scorer = ItemScorer(cfg, device, with_lora=True)
    tr = Stage1Sessions(cfg, train_sids, retrieval, labels, train=True, seed=cfg.experiment.seed + f)
    va = Stage1Sessions(cfg, val_sids, retrieval, labels, train=False)
    fdir = stage1_dir / f"fold{f}"
    train_model(cfg, scorer, tr, va, fdir)

    bs = cfg.stage1.train.eval_batch_sessions
    res = {"val": (val_sids, *predict(scorer, va, bs, return_bin=True))}
    for split, sids in eval_sids.items():
        res[split] = (sids, *predict(scorer, Stage1Sessions(cfg, sids, retrieval, None), bs, return_bin=True))
    np.savez(fdir / "probs.npz", **{f"{k}_sids": np.array(v[0]) for k, v in res.items()},
             **{f"{k}_probs": v[1] for k, v in res.items()},
             **{f"{k}_bin": v[2] for k, v in res.items() if v[2] is not None})
    del scorer
    torch.cuda.empty_cache()


def assemble_oof(stage1_dir: Path, n_folds: int, eval_splits: list[str]) -> Path:
    """Train: each session's OOF probs. Dev/test: mean of the fold models' probs."""
    sids, split, probs = [], [], []
    acc: dict[str, list[np.ndarray]] = {s: [] for s in eval_splits}
    eval_ids: dict[str, np.ndarray] = {}
    for f in range(n_folds):
        d = np.load(stage1_dir / f"fold{f}" / "probs.npz")
        sids += d["val_sids"].tolist()
        split += ["train"] * len(d["val_sids"])
        probs.append(d["val_probs"])
        for s in eval_splits:
            eval_ids[s] = d[f"{s}_sids"]
            acc[s].append(d[f"{s}_probs"])
    for s in eval_splits:
        sids += eval_ids[s].tolist()
        split += [s] * len(eval_ids[s])
        probs.append(np.mean(acc[s], axis=0))
    path = stage1_dir / "teacher_oof.npz"
    save_teacher(path, sids, split, np.concatenate(probs))
    return path


def run_full_fit(cfg: Cfg, train_sids, eval_sids, retrieval, labels, stage1_dir: Path, device: str,
                 n_epochs: int) -> Path:
    """Ablation A8: one model on all train sessions, predicting train in-sample (non-OOF teacher)."""
    set_seed(cfg.experiment.seed)
    scorer = ItemScorer(cfg, device, with_lora=True)
    tr = Stage1Sessions(cfg, train_sids, retrieval, labels, train=True, seed=cfg.experiment.seed)
    train_model(cfg, scorer, tr, None, stage1_dir / "full_fit", fixed_epochs=n_epochs)
    bs = cfg.stage1.train.eval_batch_sessions
    all_sids = {"train": train_sids, **eval_sids}
    sids, split, probs = [], [], []
    for s, ids in all_sids.items():
        sids += list(ids)
        split += [s] * len(ids)
        probs.append(predict(scorer, Stage1Sessions(cfg, ids, retrieval, None), bs))
    path = stage1_dir / "teacher_in_sample.npz"
    save_teacher(path, sids, split, np.concatenate(probs))
    return path


def run_zero_shot(cfg: Cfg, all_sids: dict[str, list[int]], retrieval, stage1_dir: Path, device: str) -> Path:
    """Baseline S1-B0 / A1: same prompt, no training."""
    scorer = ItemScorer(cfg, device, with_lora=False)
    sids, split, probs = [], [], []
    for s, ids in all_sids.items():
        sids += list(ids)
        split += [s] * len(ids)
        probs.append(predict(scorer, Stage1Sessions(cfg, ids, retrieval, None), cfg.stage1.train.eval_batch_sessions))
    path = stage1_dir / "teacher_zeroshot.npz"
    save_teacher(path, sids, split, np.concatenate(probs))
    return path
