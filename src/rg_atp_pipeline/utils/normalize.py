"""Normalization helpers for norms catalog resolution."""

from __future__ import annotations

import re


_PUNCT_TO_SPACE = str.maketrans({".": " ", ",": " ", ";": " "})


def normalize_text(value: str) -> str:
    """Normalize text for soft matching (trim, strip punctuation, uppercase)."""
    cleaned = value.translate(_PUNCT_TO_SPACE)
    return re.sub(r"\s+", " ", cleaned.strip()).upper()
