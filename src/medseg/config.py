from __future__ import annotations

import os
from copy import deepcopy

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for key, value in b.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(cfg_path: str) -> dict:
    def _load(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    base = _load(cfg_path)
    inherits = base.get("inherits", [])
    if not inherits:
        return base

    merged: dict = {}
    for parent in inherits:
        parent_path = os.path.normpath(os.path.join(os.path.dirname(cfg_path), parent))
        merged = deep_merge(merged, _load(parent_path))

    merged = deep_merge(merged, {k: v for k, v in base.items() if k != "inherits"})
    return merged
