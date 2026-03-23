"""Configuration and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def config_root() -> Path:
    """Return the Hydra config root."""

    return repo_root() / "configs"


def load_experiment_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load an experiment config from Hydra config groups."""

    with initialize_config_dir(version_base=None, config_dir=str(config_root())):
        cfg = compose(config_name=f"experiment/{config_name}", overrides=overrides or [])
    return cfg.experiment if "experiment" in cfg else cfg


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(config: DictConfig | dict[str, Any], path: Path) -> None:
    """Save a Hydra config or dict to YAML."""

    path.write_text(OmegaConf.to_yaml(config), encoding="utf-8")


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a JSON payload with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_container(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert Hydra config objects into plain dictionaries."""

    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return config
