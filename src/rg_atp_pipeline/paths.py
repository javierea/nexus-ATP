"""Project path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory based on package location."""
    override = os.environ.get("RG_ATP_PIPELINE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the data directory path."""
    return repo_root() / "data"


def config_path() -> Path:
    """Return the default config.yml path."""
    return repo_root() / "config.yml"


def state_path() -> Path:
    """Return the default state.json path."""
    return data_dir() / "state" / "state.json"
