"""Configuration management and deterministic reproducibility utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def load_config(config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML configuration file and apply optional dictionary overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if overrides:
        _deep_update(cfg, overrides)

    return cfg


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_amp_dtype(dtype_str: str = "bfloat16") -> torch.dtype:
    """Resolve automatic mixed precision dtype with hardware support check."""
    if dtype_str == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32
