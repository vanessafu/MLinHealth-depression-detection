"""YAML config loading with `_base_` inheritance and dotted CLI overrides."""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    """dict with attribute access: cfg.stage2.loss.w_tired."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base is not None:
        data = _deep_merge(_load_yaml((path.parent / base).resolve()), data)
    return data


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` overrides; value is parsed as YAML (numbers, lists, null...)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path = "config/default.yaml", overrides: list[str] | None = None) -> Cfg:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = apply_overrides(_load_yaml(path), overrides)
    return Cfg.wrap(cfg)


def resolve(p: str | Path) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def exp_dir(cfg: Cfg, *sub: str) -> Path:
    d = resolve(cfg.paths.outputs) / cfg.experiment.name
    for s in sub:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg: Cfg, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="覆盖配置，例如 --set stage2.loss.w_tired=0.25")
    return ap


def config_from_args(args: argparse.Namespace) -> Cfg:
    return load_config(args.config, args.set)
