"""State model and helpers.

Data contract:
- schema_version: integer version for future migrations
- last_run_at: ISO timestamp string or null when never run
- last_seen_n_by_year: mapping year -> last seen N (string keys)
- notes: freeform notes for operators
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class State(BaseModel):
    """Persistent pipeline state."""

    schema_version: int = Field(..., ge=1)
    last_run_at: Optional[datetime]
    last_seen_n_by_year: Dict[str, int]
    notes: str


def default_state() -> State:
    """Return default state for Etapa 0."""
    return State(
        schema_version=1,
        last_run_at=None,
        last_seen_n_by_year={},
        notes="Estado inicial",
    )


def load_state(path: Path) -> State:
    """Load and validate state.json from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return State.model_validate(data)


def save_state(state: State, path: Path) -> None:
    """Save state.json to disk."""
    payload = state.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
