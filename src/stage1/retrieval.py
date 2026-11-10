"""Per-item evidence retrieval: question map (r=1) + frozen bge cosine similarity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .. import N_ITEMS
from ..config import Cfg, resolve
from ..preprocessing.pipeline import load_qa_pairs


def load_item_defs(cfg: Cfg) -> dict[int, dict]:
    with open(resolve(cfg.stage1.retrieval.item_queries)) as f:
        return {int(k): v for k, v in yaml.safe_load(f).items()}


def build_retrieval(cfg: Cfg, sids: list[int], device: str) -> dict:
    """Returns {sid: {k: {"ranked": [pair_id...], "scores": [...], "max_r": float, "ev": 0/1}}}."""
    from sentence_transformers import SentenceTransformer

    rc = cfg.stage1.retrieval
    enc = SentenceTransformer(rc.encoder, device=device)
    items = load_item_defs(cfg)
    q_emb = {k: enc.encode([rc.query_instruction + q for q in items[k]["queries"]],
                           normalize_embeddings=True) for k in range(1, N_ITEMS + 1)}
    out = {}
    for sid in sids:
        pairs = load_qa_pairs(cfg, sid)
        if not pairs:
            out[sid] = {k: {"ranked": [], "scores": [], "max_r": -1.0, "ev": 0} for k in range(1, N_ITEMS + 1)}
            continue
        texts = [f"Q: {p['question']} A: {p['answer']}" for p in pairs]
        p_emb = enc.encode(texts, normalize_embeddings=True, batch_size=rc.batch_size)
        res = {}
        for k in range(1, N_ITEMS + 1):
            r = (p_emb @ q_emb[k].T).max(axis=1)                       # r_ijk
            if rc.use_question_map:
                r = np.where([k in p["items"] for p in pairs], 1.0, r)
            order = np.argsort(-r)[: rc.store_top]
            res[k] = {"ranked": [int(pairs[j]["pair_id"]) for j in order],
                      "scores": [round(float(r[j]), 4) for j in order],
                      "max_r": float(r.max()), "ev": int(r.max() >= rc.evidence_threshold)}
        out[sid] = res
    return out


def save_retrieval(ret: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump({str(s): {str(k): v for k, v in d.items()} for s, d in ret.items()}, f)


def load_retrieval(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(s): {int(k): v for k, v in d.items()} for s, d in raw.items()}
