"""Step 9: frozen WavLM-large; per segment, mean+std pool each layer -> [M, L, 2*1024] fp16."""
from __future__ import annotations

import numpy as np
import torch

from ..config import Cfg

_CACHE: dict = {}


def _load(cfg: Cfg, device: str):
    if "model" not in _CACHE:
        from transformers import AutoFeatureExtractor, WavLMModel

        name = cfg.preprocessing.wavlm.model
        _CACHE["fe"] = AutoFeatureExtractor.from_pretrained(name)
        _CACHE["model"] = WavLMModel.from_pretrained(name).to(device).eval()
    return _CACHE["fe"], _CACHE["model"]


@torch.no_grad()
def extract(wav: np.ndarray, segments: list[dict], cfg: Cfg, device: str) -> dict:
    fe, model = _load(cfg, device)
    sr = cfg.preprocessing.sample_rate
    layers = cfg.preprocessing.wavlm.layers
    feats = []
    for seg in segments:
        x = wav[int(seg["start"] * sr): int(seg["end"] * sr)]
        inp = fe(x, sampling_rate=sr, return_tensors="pt").input_values.to(device)
        hs = model(inp, output_hidden_states=True).hidden_states  # tuple of 25 x [1, T, 1024]
        h = torch.stack(hs, dim=0)[:, 0]                          # [25, T, 1024]
        if layers != "all":
            h = h[list(layers)]
        pooled = torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False)], dim=-1)  # [L, 2048]
        feats.append(pooled.to(torch.float16).cpu())
    n_layers = model.config.num_hidden_layers + 1 if layers == "all" else len(layers)
    dim = 2 * model.config.hidden_size
    x = torch.stack(feats) if feats else torch.zeros(0, n_layers, dim, dtype=torch.float16)
    return {
        "feats": x,
        "seg_ids": torch.tensor([s["seg_id"] for s in segments], dtype=torch.long),
        "pos": torch.tensor([s["pos"] for s in segments], dtype=torch.float32),
        "layers": layers,
    }
