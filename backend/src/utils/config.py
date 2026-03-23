import os
import yaml
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_cache = {}


def load_config(name: str) -> dict:
    if name in _cache:
        return _cache[name]

    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _cache[name] = cfg
    return cfg


def get_device(preference: str = "auto"):
    import torch
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def get_nested(cfg: dict, key: str, default: Any = None) -> Any:
    keys = key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val
