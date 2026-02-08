"""LLM review helpers for compendio missing downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class MissingDownloadCandidate:
    doc_key: str
    raw_reference: str
    evidence_snippet: str
    page_number: int | None
    status: str
    url: str
    last_checked_at: str | None
    last_downloaded_at: str | None


@dataclass(frozen=True)
class MissingDownloadReview:
    doc_key: str
    verdict: str
    org_guess: str
    confidence: float
    reason: str
    status: str
    url: str
    last_checked_at: str | None
    last_downloaded_at: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_key": self.doc_key,
            "verdict": self.verdict,
            "org_guess": self.org_guess,
            "confidence": self.confidence,
            "reason": self.reason,
            "status": self.status,
            "url": self.url,
            "last_checked_at": self.last_checked_at,
            "last_downloaded_at": self.last_downloaded_at,
        }


class LLMReviewer(Protocol):
    def review(
        self, items: Sequence[MissingDownloadCandidate]
    ) -> list[MissingDownloadReview]:
        """Review missing downloads and return verdicts."""


class FakeReviewer:
    """Deterministic reviewer for tests."""

    def __init__(self, verdicts: dict[str, tuple[str, float]] | None = None) -> None:
        self._verdicts = verdicts or {}

    def review(
        self, items: Sequence[MissingDownloadCandidate]
    ) -> list[MissingDownloadReview]:
        results: list[MissingDownloadReview] = []
        for idx, item in enumerate(items):
            verdict, confidence = self._verdicts.get(
                item.doc_key,
                ("ATP_MISSING" if idx % 2 == 0 else "OTHER_ORG", 0.9),
            )
            results.append(
                MissingDownloadReview(
                    doc_key=item.doc_key,
                    verdict=verdict,
                    org_guess="Unknown",
                    confidence=confidence,
                    reason="fake-review",
                    status=item.status,
                    url=item.url,
                    last_checked_at=item.last_checked_at,
                    last_downloaded_at=item.last_downloaded_at,
                )
            )
        return results


def filter_missing_downloads_atp(
    reviews: Sequence[MissingDownloadReview],
    confidence_threshold: float,
) -> list[MissingDownloadReview]:
    return [
        review
        for review in reviews
        if review.verdict == "ATP_MISSING" and review.confidence >= confidence_threshold
    ]
