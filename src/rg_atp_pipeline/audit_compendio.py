"""Audit compendio legislativo for missing RG references."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


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
class AuditSummary:
    run_id: str
    pdf_path: str
    export_dir: str
    min_confidence: float
    needs_ocr_compendio: bool
    total_refs_detected: int
    unique_refs_detected: int
    present_downloaded: list[str]
    present_not_downloaded: list[str]
    not_registered: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "pdf_path": self.pdf_path,
            "export_dir": self.export_dir,
            "min_confidence": self.min_confidence,
            "needs_ocr_compendio": self.needs_ocr_compendio,
            "total_refs_detected": self.total_refs_detected,
            "unique_refs_detected": self.unique_refs_detected,
            "present_downloaded": self.present_downloaded,
            "present_not_downloaded": self.present_not_downloaded,
            "not_registered": self.not_registered,
            "counts": {
                "present_downloaded": len(self.present_downloaded),
                "present_not_downloaded": len(self.present_not_downloaded),
                "not_registered": len(self.not_registered),
            },
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
    refs_csv = export_dir / f"refs_detected_{run_id}.csv"
    summary_json = export_dir / f"audit_summary_{run_id}.json"

    _write_refs_csv(refs_csv, filtered)
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
) -> tuple[list[str], list[str], list[str]]:
    if not doc_keys:
        return [], [], []

    present_downloaded: list[str] = []
    present_not_downloaded: list[str] = []
    found: set[str] = set()

    with sqlite3.connect(db_path) as conn:
        for chunk in _chunks(doc_keys, 900):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT doc_key, status, latest_pdf_path "
                f"FROM documents WHERE doc_key IN ({placeholders})"
            )
            rows = conn.execute(query, list(chunk)).fetchall()
            for doc_key, status, latest_pdf_path in rows:
                found.add(doc_key)
                if status == "DOWNLOADED" and latest_pdf_path:
                    present_downloaded.append(doc_key)
                else:
                    present_not_downloaded.append(doc_key)

    not_registered = [key for key in doc_keys if key not in found]
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
