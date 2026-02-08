"""Audit compendio legislativo for missing RG references."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pypdf import PdfReader

from .llm_review import (
    LLMReviewer,
    MissingDownloadCandidate,
    MissingDownloadReview,
    filter_missing_downloads_atp,
)


@dataclass(frozen=True)
class CompendioReference:
    raw_reference: str
    doc_key_normalized: str
    year: int | None
    number: int | None
    page_number: int
    evidence_snippet: str
    confidence: float


@dataclass(frozen=True)
class MissingDownload:
    doc_key: str
    status: str
    last_checked_at: str | None
    last_downloaded_at: str | None
    url: str

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_key": self.doc_key,
            "status": self.status,
            "last_checked_at": self.last_checked_at,
            "last_downloaded_at": self.last_downloaded_at,
            "url": self.url,
        }


@dataclass(frozen=True)
class AuditSummary:
    run_id: str
    pdf_path: str
    export_dir: str
    min_confidence: float
    needs_ocr_compendio: bool
    total_refs_detected: int
    unique_refs_detected: int
    present_downloaded: list[str]
    present_not_downloaded: list[MissingDownload]
    not_registered: list[str]
    missing_downloads_reviewed_counts: dict[str, int] | None = None
    missing_downloads_atp: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        missing_downloads = [item.as_dict() for item in self.present_not_downloaded]
        return {
            "run_id": self.run_id,
            "pdf_path": self.pdf_path,
            "export_dir": self.export_dir,
            "min_confidence": self.min_confidence,
            "needs_ocr_compendio": self.needs_ocr_compendio,
            "total_refs_detected": self.total_refs_detected,
            "unique_refs_detected": self.unique_refs_detected,
            "present_downloaded": self.present_downloaded,
            "present_not_downloaded": missing_downloads,
            "missing_downloads": missing_downloads,
            "not_registered": self.not_registered,
            "present_downloaded_count": len(self.present_downloaded),
            "present_not_downloaded_count": len(self.present_not_downloaded),
            "not_registered_count": len(self.not_registered),
            "counts": {
                "present_downloaded": len(self.present_downloaded),
                "present_not_downloaded": len(self.present_not_downloaded),
                "not_registered": len(self.not_registered),
            },
            "missing_downloads_reviewed_counts": self.missing_downloads_reviewed_counts,
            "missing_downloads_atp": self.missing_downloads_atp,
        }


NEW_RES_RE = re.compile(
    r"(?i)RES\s*[-\s]*?(\d{4})\s*[-\s]+(\d{1,4})\s*[-\s]+20\s*[-\s]+1"
)
OLD_RG_RE = re.compile(
    r"(?i)(?:RG|Resoluci[oó]n\s+General)\s*(?:N[°oº]?|No\.?|Nro\.?|N\u00ba|N\u00b0|"
    r"N\u00ba\.|N\u00b0\.|N\u00bamero|N\u00famero|Numero)?\s*[:\-]??\s*(\d{1,5})"
)
CONTEXT_RG_RE = re.compile(r"(?i)(RG|Resoluci[oó]n\s+General)")
NUMBER_RE = re.compile(r"\b(\d{1,5})\b")


def extract_compendio_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def detect_references_in_text(text: str, page_number: int = 1) -> list[CompendioReference]:
    references: list[CompendioReference] = []
    seen_keys: set[str] = set()

    for match in NEW_RES_RE.finditer(text):
        year = int(match.group(1))
        number = int(match.group(2))
        normalized = f"RES-{year}-{number}-20-1"
        references.append(
            CompendioReference(
                raw_reference=match.group(0),
                doc_key_normalized=normalized,
                year=year,
                number=number,
                page_number=page_number,
                evidence_snippet=_snippet(text, match.start(), match.end()),
                confidence=1.0,
            )
        )
        seen_keys.add(normalized)

    for match in OLD_RG_RE.finditer(text):
        number = int(match.group(1))
        normalized = f"OLD-{number}"
        if normalized in seen_keys:
            continue
        references.append(
            CompendioReference(
                raw_reference=match.group(0),
                doc_key_normalized=normalized,
                year=None,
                number=number,
                page_number=page_number,
                evidence_snippet=_snippet(text, match.start(), match.end()),
                confidence=0.85,
            )
        )
        seen_keys.add(normalized)

    for match in NUMBER_RE.finditer(text):
        number = int(match.group(1))
        normalized = f"OLD-{number}"
        if normalized in seen_keys:
            continue
        window = text[max(0, match.start() - 40) : match.end() + 40]
        if not CONTEXT_RG_RE.search(window):
            continue
        references.append(
            CompendioReference(
                raw_reference=match.group(0),
                doc_key_normalized=normalized,
                year=None,
                number=number,
                page_number=page_number,
                evidence_snippet=_snippet(text, match.start(), match.end()),
                confidence=0.6,
            )
        )
        seen_keys.add(normalized)

    return references


def run_audit_compendio(
    pdf_path: Path,
    db_path: Path,
    export_dir: Path,
    min_confidence: float = 0.0,
    save_to_db: bool = True,
    export_refs: bool = True,
    export_missing_downloads: bool = True,
) -> tuple[list[CompendioReference], AuditSummary]:
    pages = extract_compendio_pages(pdf_path)
    if not any(page.strip() for page in pages):
        summary = AuditSummary(
            run_id=_timestamp(),
            pdf_path=str(pdf_path),
            export_dir=str(export_dir),
            min_confidence=min_confidence,
            needs_ocr_compendio=True,
            total_refs_detected=0,
            unique_refs_detected=0,
            present_downloaded=[],
            present_not_downloaded=[],
            not_registered=[],
        )
        return [], summary

    detected: list[CompendioReference] = []
    for idx, page in enumerate(pages, start=1):
        detected.extend(detect_references_in_text(page, page_number=idx))

    filtered = [ref for ref in detected if ref.confidence >= min_confidence]
    unique_keys = _unique_keys(filtered)
    present_downloaded, present_not_downloaded, not_registered = _compare_to_sqlite(
        unique_keys, db_path
    )

    export_dir.mkdir(parents=True, exist_ok=True)
    run_id = _timestamp()
    summary_json = export_dir / f"audit_summary_{run_id}.json"

    if export_refs:
        refs_csv = export_dir / f"refs_detected_{run_id}.csv"
        _write_refs_csv(refs_csv, filtered)
    if export_missing_downloads:
        _write_missing_downloads_csv(
            export_dir / f"missing_downloads_{run_id}.csv", present_not_downloaded
        )
    summary = AuditSummary(
        run_id=run_id,
        pdf_path=str(pdf_path),
        export_dir=str(export_dir),
        min_confidence=min_confidence,
        needs_ocr_compendio=False,
        total_refs_detected=len(filtered),
        unique_refs_detected=len(unique_keys),
        present_downloaded=present_downloaded,
        present_not_downloaded=present_not_downloaded,
        not_registered=not_registered,
    )
    summary_json.write_text(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))

    if save_to_db:
        _save_refs_to_db(db_path, run_id, pdf_path, filtered)

    return filtered, summary


def _unique_keys(references: Iterable[CompendioReference]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in references:
        if ref.doc_key_normalized in seen:
            continue
        seen.add(ref.doc_key_normalized)
        ordered.append(ref.doc_key_normalized)
    return ordered


def _compare_to_sqlite(
    doc_keys: list[str],
    db_path: Path,
) -> tuple[list[str], list[MissingDownload], list[str]]:
    if not doc_keys:
        return [], [], []

    present_downloaded: list[str] = []
    present_not_downloaded: list[MissingDownload] = []
    found: dict[str, tuple[str, str | None, str, str | None]] = {}

    with sqlite3.connect(db_path) as conn:
        for chunk in _chunks(doc_keys, 900):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT doc_key, status, latest_pdf_path, url, last_checked_at, last_downloaded_at "
                f"FROM documents WHERE doc_key IN ({placeholders})"
            )
            rows = conn.execute(query, list(chunk)).fetchall()
            for doc_key, status, latest_pdf_path, url, last_checked_at, last_downloaded_at in rows:
                found[doc_key] = (
                    status,
                    latest_pdf_path,
                    url,
                    last_checked_at,
                    last_downloaded_at,
                )

    not_registered: list[str] = []
    for key in doc_keys:
        if key not in found:
            not_registered.append(key)
            continue
        status, latest_pdf_path, url, last_checked_at, last_downloaded_at = found[key]
        if status == "DOWNLOADED" and latest_pdf_path:
            present_downloaded.append(key)
        else:
            present_not_downloaded.append(
                MissingDownload(
                    doc_key=key,
                    status=status,
                    last_checked_at=last_checked_at,
                    last_downloaded_at=last_downloaded_at,
                    url=url,
                )
            )
    return present_downloaded, present_not_downloaded, not_registered


def _save_refs_to_db(
    db_path: Path,
    run_id: str,
    pdf_path: Path,
    references: list[CompendioReference],
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS compendio_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                doc_key_normalized TEXT NOT NULL,
                year INTEGER,
                number INTEGER,
                page_number INTEGER NOT NULL,
                confidence REAL NOT NULL,
                raw_reference TEXT NOT NULL,
                evidence_snippet TEXT NOT NULL
            )
            """
        )
        created_at = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO compendio_refs (
                run_id,
                created_at,
                pdf_path,
                doc_key_normalized,
                year,
                number,
                page_number,
                confidence,
                raw_reference,
                evidence_snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    created_at,
                    str(pdf_path),
                    ref.doc_key_normalized,
                    ref.year,
                    ref.number,
                    ref.page_number,
                    ref.confidence,
                    ref.raw_reference,
                    ref.evidence_snippet,
                )
                for ref in references
            ],
        )
        conn.commit()


def review_missing_downloads(
    missing_downloads: Sequence[MissingDownload],
    references: Sequence[CompendioReference],
    reviewer: LLMReviewer,
    export_dir: Path,
    run_id: str,
    model_name: str,
    confidence_threshold: float = 0.8,
    db_path: Path | None = None,
    save_to_db: bool = True,
) -> tuple[list[MissingDownloadReview], list[MissingDownloadReview]]:
    if not missing_downloads:
        reviewed: list[MissingDownloadReview] = []
        atp_only: list[MissingDownloadReview] = []
    else:
        candidates = _build_review_candidates(missing_downloads, references)
        reviewed = reviewer.review(candidates)
        atp_only = filter_missing_downloads_atp(reviewed, confidence_threshold)

    export_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = export_dir / "missing_downloads_reviewed.csv"
    atp_path = export_dir / "missing_downloads_atp.csv"
    _write_reviewed_csv(reviewed_path, reviewed)
    _write_reviewed_csv(atp_path, atp_only)

    if reviewed and save_to_db and db_path is not None:
        _save_missing_reviews_to_db(db_path, run_id, model_name, reviewed)

    return reviewed, atp_only


def update_audit_summary_with_review(
    summary: AuditSummary,
    reviewed: Sequence[MissingDownloadReview],
    atp_only: Sequence[MissingDownloadReview],
) -> AuditSummary:
    counts: dict[str, int] = {}
    for item in reviewed:
        counts[item.verdict] = counts.get(item.verdict, 0) + 1
    updated = replace(
        summary,
        missing_downloads_reviewed_counts=counts,
        missing_downloads_atp=[item.doc_key for item in atp_only],
    )
    summary_path = Path(summary.export_dir) / f"audit_summary_{summary.run_id}.json"
    summary_path.write_text(json.dumps(updated.as_dict(), indent=2, ensure_ascii=False))
    return updated


def _build_review_candidates(
    missing_downloads: Sequence[MissingDownload],
    references: Sequence[CompendioReference],
) -> list[MissingDownloadCandidate]:
    by_doc: dict[str, CompendioReference] = {}
    for ref in references:
        existing = by_doc.get(ref.doc_key_normalized)
        if existing is None or ref.confidence > existing.confidence:
            by_doc[ref.doc_key_normalized] = ref

    candidates: list[MissingDownloadCandidate] = []
    for item in missing_downloads:
        ref = by_doc.get(item.doc_key)
        candidates.append(
            MissingDownloadCandidate(
                doc_key=item.doc_key,
                raw_reference=ref.raw_reference if ref else "",
                evidence_snippet=ref.evidence_snippet if ref else "",
                page_number=ref.page_number if ref else None,
                status=item.status,
                url=item.url,
                last_checked_at=item.last_checked_at,
                last_downloaded_at=item.last_downloaded_at,
            )
        )
    return candidates


def _write_missing_downloads_csv(path: Path, missing: Sequence[MissingDownload]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["doc_key", "status", "last_checked_at", "last_downloaded_at", "url"]
        )
        for item in missing:
            writer.writerow(
                [
                    item.doc_key,
                    item.status,
                    item.last_checked_at,
                    item.last_downloaded_at,
                    item.url,
                ]
            )


def _write_reviewed_csv(path: Path, reviews: Sequence[MissingDownloadReview]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "doc_key",
                "verdict",
                "org_guess",
                "confidence",
                "reason",
                "status",
                "url",
                "last_checked_at",
                "last_downloaded_at",
            ]
        )
        for review in reviews:
            writer.writerow(
                [
                    review.doc_key,
                    review.verdict,
                    review.org_guess,
                    f"{review.confidence:.2f}",
                    review.reason,
                    review.status,
                    review.url,
                    review.last_checked_at,
                    review.last_downloaded_at,
                ]
            )


def _save_missing_reviews_to_db(
    db_path: Path,
    run_id: str,
    model_name: str,
    reviews: Sequence[MissingDownloadReview],
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS compendio_missing_review (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                doc_key TEXT NOT NULL,
                model TEXT NOT NULL,
                verdict TEXT NOT NULL,
                org_guess TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        created_at = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO compendio_missing_review (
                run_id,
                created_at,
                doc_key,
                model,
                verdict,
                org_guess,
                confidence,
                reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    created_at,
                    review.doc_key,
                    model_name,
                    review.verdict,
                    review.org_guess,
                    review.confidence,
                    review.reason,
                )
                for review in reviews
            ],
        )
        conn.commit()


def _write_refs_csv(path: Path, references: list[CompendioReference]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "doc_key_normalized",
                "year",
                "number",
                "page_number",
                "confidence",
                "raw_reference",
                "evidence_snippet",
            ]
        )
        for ref in references:
            writer.writerow(
                [
                    ref.doc_key_normalized,
                    ref.year,
                    ref.number,
                    ref.page_number,
                    f"{ref.confidence:.2f}",
                    ref.raw_reference,
                    ref.evidence_snippet,
                ]
            )


def _snippet(text: str, start: int, end: int, max_len: int = 200) -> str:
    half = max_len // 2
    left = max(0, start - half)
    right = min(len(text), end + half)
    snippet = text[left:right].replace("\n", " ").strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip()
    return snippet


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
