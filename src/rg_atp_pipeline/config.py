"""Configuration model and helpers.

Data contract:
- base_url_new: base URL for new PDFs (placeholder at Etapa 0)
- base_url_old: base URL for old PDFs (placeholder at Etapa 0)
- rate_limit_rps: max requests per second (future use)
- user_agent: User-Agent header for HTTP requests
- years: list of years to process
- max_n_by_year: mapping year -> max N to attempt (string keys)
- old_range: inclusive range for old PDFs (start, end)
- verify_last_k: how many latest entries to verify (future use)
- request_timeout_sec: timeout in seconds for HTTP requests
- retry: retry policy (max attempts, backoff seconds)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


class OldRange(BaseModel):
    """Range for legacy PDFs."""

    start: int = Field(..., ge=1)
    end: int = Field(..., ge=1)


class RetryPolicy(BaseModel):
    """Retry policy for HTTP requests."""

    max_attempts: int = Field(..., ge=1)
    backoff_sec: int = Field(..., ge=0)


class Config(BaseModel):
    """Root configuration model for rg_atp_pipeline."""

    base_url_new: str
    base_url_old: str
    rate_limit_rps: int = Field(..., ge=1)
    user_agent: str
    years: List[int]
    max_n_by_year: Dict[str, int]
    old_range: OldRange
    old_min_number: int = Field(..., ge=1)
    old_year_start: int = Field(..., ge=2000, le=2100)
    old_year_end: int = Field(..., ge=2000, le=2100)
    verify_last_k: int = Field(..., ge=0)
    request_timeout_sec: int = Field(..., ge=1)
    retry: RetryPolicy


def default_config() -> Config:
    """Return default configuration values for Etapa 0."""
    return Config(
        base_url_new="https://atp.chaco.gob.ar/documentos/legislativos/resoluciones-generales",
        base_url_old="https://atp.chaco.gob.ar/documentos/legislativos/resoluciones-generales",
        rate_limit_rps=1,
        user_agent="rg_atp_pipeline/0.1",
        years=[2026, 2025, 2024],
        max_n_by_year={"2026": 7, "2025": 43, "2024": 39},
        old_range=OldRange(start=1, end=2172),
        old_min_number=1195,
        old_year_start=2004,
        old_year_end=2023,
        verify_last_k=5,
        request_timeout_sec=30,
        retry=RetryPolicy(max_attempts=3, backoff_sec=2),
    )


def load_config(path: Path) -> Config:
    """Load and validate config.yml from disk."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config.model_validate(data)


def save_config(config: Config, path: Path) -> None:
    """Save config.yml to disk."""
    payload = config.model_dump()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
