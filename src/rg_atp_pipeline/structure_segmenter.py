"""Deterministic structural segmentation for rg_atp_pipeline (Etapa 3)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .storage_sqlite import DocStructureRecord, DocumentStore, UnitRecord

SECTION_HEADER_RE = re.compile(
    r"^\s*(VISTO|CONSIDERANDO|RESUELVE|RESUELVEN|RESUELVO)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ARTICLE_RE = re.compile(
    r"^\s*(?:ART(?:[ÍI]CULO)?|ART\.)\s+(\d+\s*(?:[º°])?)",
    re.IGNORECASE | re.MULTILINE,
)
ANNEX_RE = re.compile(
    r"^\s*ANEXO(?:\s+([A-Z0-9IVXLCDMÑÚÜº°\-]+|ÚNICO|UNICO))?\s*[:.\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class StructureOptions:
    doc_key: str | None
    limit: int | None
    force: bool
    include_needs_ocr: bool
    export_json: bool


@dataclass(frozen=True)
class StructureSummary:
    processed: int
    structured_ok: int
    partial: int
    skipped: int
    error: int
    total_articles: int

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "structured_ok": self.structured_ok,
            "partial": self.partial,
            "skipped": self.skipped,
            "error": self.error,
            "total_articles": self.total_articles,
        }


@dataclass(frozen=True)
class UnitDraft:
    unit_type: str
    unit_number: str | None
    title: str | None
    text: str
    start_char: int | None
    end_char: int | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class StructuredDocument:
    units: list[UnitDraft]
    sections: dict[str, str | None]
    articles: list[dict[str, str]]
    annexes: list[dict[str, str]]
    warnings: list[str]
    confidence: float


def run_structure(
    store: DocumentStore,
    data_dir: Path,
    options: StructureOptions,
    logger: logging.Logger,
) -> StructureSummary:
    store.initialize()
    candidates = store.list_text_candidates(
        status="DOWNLOADED",
        limit=options.limit,
        doc_key=options.doc_key,
        only_text_status=None,
    )

    processed = 0
    structured_ok = 0
    partial = 0
    skipped = 0
    error = 0
    total_articles = 0

    structured_dir = data_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)

    for record in candidates:
        if record.text_status != "EXTRACTED" and not (
            options.include_needs_ocr and record.text_status == "NEEDS_OCR"
        ):
            skipped += 1
            continue

        doc_structure = store.get_doc_structure(record.doc_key)
        if (
            doc_structure
            and doc_structure.structure_status == "STRUCTURED"
            and not options.force
            and _is_up_to_date(record.text_extracted_at, doc_structure.structured_at)
        ):
            logger.info(
                "Estructura omitida %s (status=%s).",
                record.doc_key,
                doc_structure.structure_status,
            )
            skipped += 1
            continue

        processed += 1

        if not record.text_path:
            logger.warning("Sin texto para %s.", record.doc_key)
            store.upsert_doc_structure(
                DocStructureRecord(
                    doc_key=record.doc_key,
                    structure_status="ERROR",
                    structure_confidence=0.0,
                    articles_detected=0,
                    annexes_detected=0,
                    notes=json.dumps(["text_path_missing"]),
                    structured_at=_now_iso(),
                )
            )
            error += 1
            continue

        text_path = Path(record.text_path)
        if not text_path.exists():
            logger.warning("Texto no encontrado %s para %s.", text_path, record.doc_key)
            store.upsert_doc_structure(
                DocStructureRecord(
                    doc_key=record.doc_key,
                    structure_status="ERROR",
                    structure_confidence=0.0,
                    articles_detected=0,
                    annexes_detected=0,
                    notes=json.dumps(["text_file_missing"]),
                    structured_at=_now_iso(),
                )
            )
            error += 1
            continue

        raw_text = text_path.read_text(encoding="utf-8")
        structured = _segment_text(raw_text)
        articles_detected = len(structured.articles)
        annexes_detected = len(structured.annexes)
        status = "STRUCTURED" if articles_detected else "PARTIAL"

        store.delete_units(record.doc_key)
        store.insert_units(
            [
                UnitRecord(
                    id=0,
                    doc_key=record.doc_key,
                    unit_type=unit.unit_type,
                    unit_number=unit.unit_number,
                    title=unit.title,
                    text=unit.text,
                    start_char=unit.start_char,
                    end_char=unit.end_char,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    created_at=_now_iso(),
                )
                for unit in structured.units
            ]
        )
        store.upsert_doc_structure(
            DocStructureRecord(
                doc_key=record.doc_key,
                structure_status=status,
                structure_confidence=structured.confidence,
                articles_detected=articles_detected,
                annexes_detected=annexes_detected,
                notes=json.dumps(structured.warnings),
                structured_at=_now_iso(),
            )
        )

        if options.export_json:
            payload = {
                "doc_key": record.doc_key,
                "sections": structured.sections,
                "articles": structured.articles,
                "annexes": structured.annexes,
                "metrics": {
                    "articles_detected": articles_detected,
                    "confidence": structured.confidence,
                    "warnings": structured.warnings,
                },
            }
            (structured_dir / f"{record.doc_key}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        total_articles += articles_detected
        if status == "STRUCTURED":
            structured_ok += 1
        else:
            partial += 1

    return StructureSummary(
        processed=processed,
        structured_ok=structured_ok,
        partial=partial,
        skipped=skipped,
        error=error,
        total_articles=total_articles,
    )


def _segment_text(raw_text: str) -> StructuredDocument:
    sections = _extract_sections(raw_text)
    articles = _extract_articles(raw_text)
    annexes = _extract_annexes(raw_text)
    warnings = _collect_warnings(raw_text, sections, articles)

    units: list[UnitDraft] = []
    if sections["header"]:
        units.append(
            UnitDraft(
                unit_type="HEADER",
                unit_number=None,
                title=None,
                text=sections["header"],
                start_char=sections["_header_range"][0],
                end_char=sections["_header_range"][1],
                start_line=sections["_header_range"][2],
                end_line=sections["_header_range"][3],
            )
        )

    for key, unit_type in (
        ("visto", "VISTO"),
        ("considerando", "CONSIDERANDO"),
        ("resuelve", "RESUELVE"),
    ):
        if sections.get(key):
            start_char, end_char, start_line, end_line = sections[f"_{key}_range"]
            units.append(
                UnitDraft(
                    unit_type=unit_type,
                    unit_number=None,
                    title=None,
                    text=sections[key],
                    start_char=start_char,
                    end_char=end_char,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    for article in articles:
        units.append(
            UnitDraft(
                unit_type="ARTICULO",
                unit_number=article["number"],
                title=None,
                text=article["text"],
                start_char=article["_range"][0],
                end_char=article["_range"][1],
                start_line=article["_range"][2],
                end_line=article["_range"][3],
            )
        )

    for annex in annexes:
        units.append(
            UnitDraft(
                unit_type="ANEXO",
                unit_number=annex["label"],
                title=annex["title"],
                text=annex["text"],
                start_char=annex["_range"][0],
                end_char=annex["_range"][1],
                start_line=annex["_range"][2],
                end_line=annex["_range"][3],
            )
        )

    if not articles:
        trimmed, start_char, end_char = _trim_slice(raw_text, 0, len(raw_text))
        start_line, end_line = _span_lines(raw_text, start_char, end_char)
        units.append(
            UnitDraft(
                unit_type="OTRO",
                unit_number=None,
                title=None,
                text=trimmed,
                start_char=start_char,
                end_char=end_char,
                start_line=start_line,
                end_line=end_line,
            )
        )

    confidence = _compute_confidence(raw_text, sections, articles, annexes)
    cleaned_sections = {
        "header": sections["header"],
        "visto": sections.get("visto"),
        "considerando": sections.get("considerando"),
        "resuelve": sections.get("resuelve"),
    }
    return StructuredDocument(
        units=units,
        sections=cleaned_sections,
        articles=[_strip_range(item) for item in articles],
        annexes=[_strip_range(item) for item in annexes],
        warnings=warnings,
        confidence=confidence,
    )


def _extract_sections(raw_text: str) -> dict[str, str | None]:
    matches = list(SECTION_HEADER_RE.finditer(raw_text))
    line_starts = _line_starts(raw_text)
    sections: dict[str, str | None] = {
        "header": None,
        "visto": None,
        "considerando": None,
        "resuelve": None,
    }
    ranges: dict[str, tuple[int, int, int, int]] = {}

    if matches:
        first = matches[0]
        header_slice, start_char, end_char = _trim_slice(raw_text, 0, first.start())
        if header_slice:
            start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)
            sections["header"] = header_slice
            ranges["_header_range"] = (start_char, end_char, start_line, end_line)

        for idx, match in enumerate(matches):
            heading = match.group(1).strip().upper()
            key = "resuelve" if heading.startswith("RESUEL") else heading.lower()
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            slice_text, start_char, end_char = _trim_slice(raw_text, start, end)
            if slice_text:
                start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)
                sections[key] = slice_text
                ranges[f"_{key}_range"] = (start_char, end_char, start_line, end_line)
    else:
        trimmed, start_char, end_char = _trim_slice(raw_text, 0, len(raw_text))
        if trimmed:
            start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)
            sections["header"] = trimmed
            ranges["_header_range"] = (start_char, end_char, start_line, end_line)

    sections.update(ranges)
    return sections


def _extract_articles(raw_text: str) -> list[dict[str, str]]:
    matches = list(ARTICLE_RE.finditer(raw_text))
    annex_positions = [match.start() for match in ANNEX_RE.finditer(raw_text)]
    line_starts = _line_starts(raw_text)

    articles: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        next_article = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        next_annex = _next_boundary(annex_positions, start)
        end = min(next_article, next_annex) if next_annex is not None else next_article
        slice_text, start_char, end_char = _trim_slice(raw_text, start, end)
        number = match.group(1).strip()
        start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)
        articles.append(
            {
                "number": number,
                "text": slice_text,
                "_range": (start_char, end_char, start_line, end_line),
            }
        )
    return articles


def _extract_annexes(raw_text: str) -> list[dict[str, str]]:
    matches = list(ANNEX_RE.finditer(raw_text))
    line_starts = _line_starts(raw_text)
    annexes: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        slice_text, start_char, end_char = _trim_slice(raw_text, start, end)
        label = match.group(1).strip() if match.group(1) else None
        title = f"ANEXO {label}".strip() if label else "ANEXO"
        start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)
        annexes.append(
            {
                "label": label,
                "title": title,
                "text": slice_text,
                "_range": (start_char, end_char, start_line, end_line),
            }
        )
    return annexes


def _collect_warnings(
    raw_text: str,
    sections: dict[str, str | None],
    articles: list[dict[str, str]],
) -> list[str]:
    warnings: list[str] = []
    if not articles:
        warnings.append("no_articles_detected")
    if not sections.get("resuelve"):
        warnings.append("resuelve_missing")

    numbers = [article["number"] for article in articles]
    duplicates = _find_duplicates(numbers)
    if duplicates:
        warnings.append("duplicate_article_numbers")

    numeric = [_parse_article_number(value) for value in numbers]
    numeric = [value for value in numeric if value is not None]
    if numeric and not _is_monotonic_with_small_gaps(numeric):
        warnings.append("article_numbers_non_monotonic")

    if len(raw_text.strip()) < 500:
        warnings.append("very_short_document")
    return warnings


def _compute_confidence(
    raw_text: str,
    sections: dict[str, str | None],
    articles: list[dict[str, str]],
    annexes: list[dict[str, str]],
) -> float:
    score = 0.3
    if sections.get("resuelve"):
        score += 0.2
    if articles:
        score += 0.2
    numbers = [_parse_article_number(article["number"]) for article in articles]
    numbers = [value for value in numbers if value is not None]
    if numbers and _is_monotonic_with_small_gaps(numbers):
        score += 0.1
    if sections.get("visto") or sections.get("considerando"):
        score += 0.1
    if re.search(r"\bANEXO\b", raw_text, re.IGNORECASE) and annexes:
        score += 0.1
    return max(0.0, min(1.0, score))


def _trim_slice(raw_text: str, start: int, end: int) -> tuple[str, int, int]:
    slice_text = raw_text[start:end]
    left_trimmed = slice_text.lstrip()
    left_delta = len(slice_text) - len(left_trimmed)
    right_trimmed = left_trimmed.rstrip()
    right_delta = len(left_trimmed) - len(right_trimmed)
    new_start = start + left_delta
    new_end = end - right_delta
    return right_trimmed, new_start, new_end


def _line_starts(raw_text: str) -> list[int]:
    starts = [0]
    for idx, char in enumerate(raw_text):
        if char == "\n":
            starts.append(idx + 1)
    return starts


def _span_lines(
    raw_text: str,
    start: int,
    end: int,
    line_starts: list[int] | None = None,
) -> tuple[int | None, int | None]:
    if start is None or end is None:
        return None, None
    if line_starts is None:
        line_starts = _line_starts(raw_text)
    start_line = _line_for_char(start, line_starts)
    end_line = _line_for_char(max(end - 1, 0), line_starts)
    return start_line, end_line


def _line_for_char(index: int, line_starts: Iterable[int]) -> int:
    starts = list(line_starts)
    low = 0
    high = len(starts) - 1
    while low <= high:
        mid = (low + high) // 2
        if starts[mid] <= index:
            low = mid + 1
        else:
            high = mid - 1
    return max(1, high + 1)


def _next_boundary(boundaries: list[int], start: int) -> int | None:
    for boundary in boundaries:
        if boundary > start:
            return boundary
    return None


def _find_duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _parse_article_number(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _is_monotonic_with_small_gaps(numbers: list[int]) -> bool:
    for prev, curr in zip(numbers, numbers[1:]):
        if curr <= prev:
            return False
        if curr - prev > 5:
            return False
    return True


def _strip_range(item: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _is_up_to_date(text_extracted_at: str | None, structured_at: str | None) -> bool:
    if not structured_at:
        return False
    if not text_extracted_at:
        return True
    try:
        return datetime.fromisoformat(structured_at) >= datetime.fromisoformat(
            text_extracted_at
        )
    except ValueError:
        return structured_at >= text_extracted_at


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
