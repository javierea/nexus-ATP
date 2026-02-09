"""Regex-based reference extraction for normativa citations."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    raw_text: str
    norm_type_guess: str
    norm_key_candidate: str | None
    regex_confidence: float
    span_start: int | None
    span_end: int | None
    evidence_snippet: str


LAW_NUMBER_RE = re.compile(r"\bLey\s*(?:N[°º]\s*)?(\d{1,4}-[A-Z])\b", re.IGNORECASE)
LAW_NAME_RE = re.compile(
    r"\b(Código\s+Tributario\s+Provincial|CTP|Ley\s+Tarifaria)\b",
    re.IGNORECASE,
)
DECREE_RE = re.compile(
    r"\bDecreto\s*(Ley\s*)?(?:N[°º]\s*)?(\d{1,4})/(\d{2,4})\b",
    re.IGNORECASE,
)
DECREE_SHORT_RE = re.compile(
    r"\bDec\.?\s*(Ley\s*)?(\d{1,4})/(\d{2,4})\b",
    re.IGNORECASE,
)
RG_ATP_KEY_RE = re.compile(r"\bRES-\d{4}-\d{1,3}-20-1\b", re.IGNORECASE)
RG_ATP_RE = re.compile(
    r"\b(?:Resolución\s+General|RG)\s*(?:N[°º]\s*)?(\d{1,4})\b",
    re.IGNORECASE,
)
RG_CA_RE = re.compile(
    r"\bRG\s*CA\s*(\d{1,3})/(\d{4})\b|\bResolución\s+General\s*\(C\.A\.\)\s*(\d{1,3})/(\d{4})\b",
    re.IGNORECASE,
)


def extract_candidates(text: str) -> list[Candidate]:
    """Extract reference candidates from text using regex heuristics."""
    results: list[Candidate] = []
    seen_spans: set[tuple[int, int, str]] = set()

    def add_candidate(
        match: re.Match[str],
        norm_type: str,
        norm_key: str | None,
        confidence: float,
    ) -> None:
        start, end = match.span()
        raw_text = match.group(0)
        key = (start, end, raw_text)
        if key in seen_spans:
            return
        seen_spans.add(key)
        evidence = _snippet(text, start, end, 300)
        results.append(
            Candidate(
                raw_text=raw_text,
                norm_type_guess=norm_type,
                norm_key_candidate=norm_key,
                regex_confidence=confidence,
                span_start=start,
                span_end=end,
                evidence_snippet=evidence,
            )
        )

    for match in RG_CA_RE.finditer(text):
        add_candidate(match, "RG_CA", None, 0.8)

    for match in RG_ATP_KEY_RE.finditer(text):
        add_candidate(match, "RG_ATP", match.group(0).upper(), 0.97)

    for match in LAW_NUMBER_RE.finditer(text):
        number = match.group(1).upper()
        add_candidate(match, "LEY", f"LEY-{number}", 0.9)

    for match in LAW_NAME_RE.finditer(text):
        confidence = 0.6
        if "TARIFARIA" in match.group(0).upper():
            confidence = 0.65
        add_candidate(match, "LEY", None, confidence)

    for match in DECREE_RE.finditer(text):
        add_candidate(match, "DECRETO", None, 0.8 if match.group(1) else 0.78)

    for match in DECREE_SHORT_RE.finditer(text):
        add_candidate(match, "DECRETO", None, 0.75)

    for match in RG_ATP_RE.finditer(text):
        raw = match.group(0)
        if "CA" in raw.upper():
            continue
        number = match.group(1)
        add_candidate(match, "RG_ATP", f"OLD-{number}", 0.85)

    return results


def _snippet(text: str, start: int, end: int, radius: int) -> str:
    begin = max(0, start - radius)
    finish = min(len(text), end + radius)
    return text[begin:finish].strip()
