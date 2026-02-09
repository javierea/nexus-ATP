"""Normalization helpers for norms catalog resolution."""

from __future__ import annotations

import re


def normalize_text(value: str) -> str:
    """Normalize text for soft matching (trim, collapse spaces, uppercase)."""
    return re.sub(r"\s+", " ", value.strip()).upper()
