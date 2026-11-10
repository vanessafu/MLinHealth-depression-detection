"""Stage 2 training: 5 folds x seeds, early stopping on val CCC, optional grid over (w_t, w_m, lambda)."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Cfg, set_seed
from ..evaluation.metrics import ccc
from .dataset import Bags
from .losses import lds_weights, stage2_loss
from .model import ResidualMIL


def build_model(cfg: Cfg, bags: Bags) -> ResidualMIL:
    mc = cfg.stage2.model
    return ResidualMIL(n_layers=bags.n_layers, feat_dim=bags.feat_dim, proj_dim=mc.proj_dim,
                       attn_dim=mc.attn_dim, hidden_dim=mc.hidden_dim, dropout=mc.dropout,
                       max_delta=mc.max_delta, pooling=mc.pooling, use_pos=mc.use_pos)


@torch.no_grad()
def predict(model: ResidualMIL, bags: Bags, idx: list[int], device: str, batch_sessions: int = 8) -> dict:
    """Inference uses all segments (no subsampling)."""
    model.eval()
    keys = ["y_tot", "tired_logits", "mov_logit", "deltas"]
    acc = {k: [] for k in keys}
    attn = []
    for i in range(0, len(idx), batch_sessions):
        b = bags.batch(idx[i: i + batch_sessions], None, None, device)
        out = model(b["feats"], b["pos"], b["mask"], b["y_text"], b["tau"], b["nu"])
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
        for a, m in zip(out["attn"].cpu().numpy(), b["mask"].cpu().numpy()):
            attn.append(a[m])
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res["attn"] = attn
    return res


def train_one(cfg: Cfg, bags: Bags, tr_idx: list[int], va_idx: list[int], seed: int, device: str):
    tc, lc = cfg.stage2.train, cfg.stage2.loss
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg, bags).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    mov_tr = bags.mov[tr_idx].numpy()
    pos_weight = torch.tensor(max(1.0, (mov_tr == 0).sum()) / max(1.0, (mov_tr == 1).sum()), device=device)
    if lc.use_lds:
        wmap = lds_weights(bags.y[tr_idx].numpy(), lc.lds_sigma)
        omega_all = torch.tensor([wmap[int(v)] for v in bags.y.numpy()], dtype=torch.float32)
    else:
        omega_all = torch.ones(len(bags))

    bs = tc.batch_sessions or len(tr_idx)
    best = {"ccc": -np.inf, "epoch": -1, "state": None}
    bad = 0
    y_va = bags.y[va_idx].numpy()
    for epoch in range(tc.max_epochs):
        model.train()
        order = rng.permutation(tr_idx).tolist()
        for s0 in range(0, len(order), bs):
            idx = order[s0: s0 + bs]
            b = bags.batch(idx, tc.max_segments, rng, device)
            out = model(b["feats"], b["pos"], b["mask"], b["y_text"], b["tau"], b["nu"])
            loss, _ = stage2_loss(out, b["y"], b["tired3"], b["mov"], omega_all[idx].to(device), pos_weight, lc)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_ccc = ccc(y_va, predict(model, bags, va_idx, device)["y_tot"])
        if val_ccc > best["ccc"]:
            best.update(ccc=val_ccc, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
            if bad >= tc.patience:
                break
    model.load_state_dict(best["state"])
    return model, best


def run_cv(cfg: Cfg, bags: Bags, folds: pd.DataFrame, eval_idx: dict[str, list[int]], device: str,
           save_dir: Path | None = None) -> dict:
    """Returns seed-averaged OOF predictions on train and fold×seed-averaged predictions on eval splits."""
    sid_to_i = {s: i for i, s in enumerate(bags.sids)}
    n_folds = int(folds.fold.max()) + 1
    train_i = [sid_to_i[s] for s in folds.sid]
    oof = {k: np.zeros((len(bags),) + ((2,) if k == "tired_logits" else ())) for k in
           ["y_tot", "tired_logits", "mov_logit"]}
    ev = {s: {k: [] for k in oof} for s in eval_idx}
    attn_ev = {s: [] for s in eval_idx}
    log = []
    seeds = list(cfg.stage2.train.seeds)
    for f in range(n_folds):
        tr = [sid_to_i[s] for s in folds.sid[folds.fold != f]]
        va = [sid_to_i[s] for s in folds.sid[folds.fold == f]]
        for seed in seeds:
            model, best = train_one(cfg, bags, tr, va, seed=seed * 1000 + f, device=device)
            log.append({"fold": f, "seed": seed, "best_epoch": best["epoch"], "val_ccc": best["ccc"]})
            p = predict(model, bags, va, device)
            for k in oof:
                oof[k][va] += p[k] / len(seeds)
            for s, idx in eval_idx.items():
                pe = predict(model, bags, idx, device)
                for k in oof:
                    ev[s][k].append(pe[k])
                attn_ev[s].append(pe["attn"])
            if save_dir is not None:
                (save_dir / "models").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "models" / f"fold{f}_seed{seed}.pt")
    y_tr = bags.y[train_i].numpy()
    res = {
        "cv_ccc": ccc(y_tr, oof["y_tot"][train_i]),
        "log": log,
        "train_idx": train_i,
        "oof": {k: v[train_i] for k, v in oof.items()},
        "eval": {s: {k: np.mean(v, axis=0) for k, v in d.items()} for s, d in ev.items()},
        # 注意力：对 25 个模型取平均（同一 session 的片段数相同）
        "eval_attn": {s: [np.mean([m[j] for m in attn_ev[s]], axis=0) for j in range(len(eval_idx[s]))]
                      for s in eval_idx},
    }
    return res


def grid_configs(cfg: Cfg):
    g = cfg.stage2.grid
    for wt, wm, lam in itertools.product(g.w_tired, g.w_mov, g.lambda_res):
        c = copy.deepcopy(cfg)
        c.stage2.loss.w_tired, c.stage2.loss.w_mov, c.stage2.loss.lambda_res = wt, wm, lam
        yield {"w_tired": wt, "w_mov": wm, "lambda_res": lam}, c


def predictions_frame(bags: Bags, idx: list[int], pred: dict, split: str) -> pd.DataFrame:
    tl = pred["tired_logits"]
    sig = 1 / (1 + np.exp(-tl))
    return pd.DataFrame({
        "sid": [bags.sids[i] for i in idx], "split": split,
        "y_text": bags.y_text[idx].numpy(), "y_tot": pred["y_tot"],
        "tired_expected": sig.sum(-1), "mov_prob": 1 / (1 + np.exp(-pred["mov_logit"])),
    })


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
