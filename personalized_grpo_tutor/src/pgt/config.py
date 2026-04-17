from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml_with_extends(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if extends is None:
        return data

    base_path = (path.parent / extends).resolve() if not Path(extends).is_absolute() else Path(extends)
    base_cfg = load_yaml_with_extends(base_path)
    return _deep_update(base_cfg, data)
