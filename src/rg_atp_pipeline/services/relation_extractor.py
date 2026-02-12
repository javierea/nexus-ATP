"""Deterministic extraction of normative relation types from text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationCandidate:
    """Candidate relation extracted from local citation context."""

    relation_type: str
    direction: str
    scope: str
    scope_detail: str | None
    confidence: float
    evidence_snippet: str
    explanation: str


_RELATION_PATTERNS: list[tuple[str, re.Pattern[str], float, str]] = [
    (
        "REPEALS",
        re.compile(
            r"\b(?:derógase|deróganse|queda\s+derogado|déjase\s+sin\s+efecto)\b",
            re.IGNORECASE,
        ),
        0.9,
        "Derogación explícita detectada.",
    ),
    (
        "SUBSTITUTES",
        re.compile(
            r"\b(?:sustitúyese|sustitúyanse|reemplázase|reemplázanse)\b",
            re.IGNORECASE,
        ),
        0.9,
        "Sustitución explícita detectada.",
    ),
    (
        "MODIFIES",
        re.compile(r"\b(?:modifícase|modifícanse|modificar)\b", re.IGNORECASE),
        0.9,
        "Modificación explícita detectada.",
    ),
    (
        "INCORPORATES",
        re.compile(r"\b(?:incorpórase|incorpóranse|agrégase|añádase)\b", re.IGNORECASE),
        0.8,
        "Incorporación explícita detectada.",
    ),
    (
        "REGULATES",
        re.compile(r"\b(?:reglaméntase|reglamenta)\b", re.IGNORECASE),
        0.8,
        "Reglamentación explícita detectada.",
    ),
    (
        "COMPLEMENTS",
        re.compile(r"\b(?:complementa|complementario)\b", re.IGNORECASE),
        0.8,
        "Relación complementaria detectada.",
    ),
    (
        "ACCORDING_TO",
        re.compile(
            r"\b(?:según|conforme(?:\s+a)?|de\s+acuerdo\s+con?|en\s+los\s+términos\s+de)\b",
            re.IGNORECASE,
        ),
        0.6,
        "Conector normativo de remisión detectado.",
    ),
]

_ARTICLE_RE = re.compile(
    r"\b(?:art\.?|artículo|articulos|artículos|arts\.?)\s*(\d+[\w°º]*)",
    re.IGNORECASE,
)
_ANNEX_RE = re.compile(
    r"\b(?:anexo|anexos)\s*([IVXLCDM]+|\d+)?\b",
    re.IGNORECASE,
)


def extract_relation_candidates(text: str) -> list[RelationCandidate]:
    """Extract deterministic relation candidates from text snippets."""
    if not text or not text.strip():
        return []

    candidates: list[RelationCandidate] = []
    normalized = " ".join(text.split())
    seen: set[tuple[str, str, str | None]] = set()

    for relation_type, pattern, confidence, explanation in _RELATION_PATTERNS:
        for match in pattern.finditer(normalized):
            candidate_scope, candidate_scope_detail = _detect_scope(
                normalized,
                match.start(),
                match.end(),
            )
            if candidate_scope == "UNKNOWN" and relation_type == "REPEALS":
                candidate_scope = "WHOLE_NORM"
            direction = "SOURCE_TO_TARGET"
            if relation_type == "ACCORDING_TO":
                direction = "UNKNOWN"

            key = (relation_type, candidate_scope, candidate_scope_detail)
            if key in seen:
                continue
            seen.add(key)

            evidence = _build_evidence(normalized, match.start(), match.end())
            candidates.append(
                RelationCandidate(
                    relation_type=relation_type,
                    direction=direction,
                    scope=candidate_scope,
                    scope_detail=candidate_scope_detail,
                    confidence=confidence,
                    evidence_snippet=evidence,
                    explanation=explanation,
                )
            )
            if len(candidates) >= 20:
                return candidates

    return candidates


def _detect_scope(text: str, start: int, end: int) -> tuple[str, str | None]:
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    scope_window = text[left:right]
    center = (start + end) // 2

    article = _nearest_match(_ARTICLE_RE, scope_window, left, center)
    if article:
        number = article.group(1).upper().replace("º", "").replace("°", "")
        return "ARTICLE", f"ART_{number}"

    annex = _nearest_match(_ANNEX_RE, scope_window, left, center)
    if annex:
        value = annex.group(1)
        detail = None
        if value:
            detail = f"ANEXO_{value.upper()}"
        return "ANNEX", detail

    return "UNKNOWN", None


def _build_evidence(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def _nearest_match(
    pattern: re.Pattern[str],
    window_text: str,
    offset: int,
    center: int,
) -> re.Match[str] | None:
    nearest: re.Match[str] | None = None
    best_distance: int | None = None
    for match in pattern.finditer(window_text):
        absolute_start = offset + match.start()
        absolute_end = offset + match.end()
        absolute_center = (absolute_start + absolute_end) // 2
        distance = abs(absolute_center - center)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            nearest = match
    return nearest
