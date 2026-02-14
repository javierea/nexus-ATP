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
    r"(?im)^(?:[ \t]*)(?:ART[ÍI]CULO|Art[íi]culo|ART\.|Art\.)\s+(\d+(?:\s*(?:bis|ter|quater|quinquies))?)\s*[º°]?\s*[:.\-]",
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
    metrics: dict[str, int | str]
    confidence: float


@dataclass(frozen=True)
class ArticleCandidate:
    number_raw: str
    number_base: int
    suffix: str | None
    header_text: str
    start_char: int
    start_line: int


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
                notes=json.dumps({"warnings": structured.warnings, "metrics": structured.metrics}, ensure_ascii=False),
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
                    **structured.metrics,
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
    article_parse = _extract_articles(raw_text)
    articles = article_parse["articles"]
    metrics = article_parse["metrics"]
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

    confidence = _compute_confidence(raw_text, sections, articles, annexes, metrics)
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
        metrics=metrics,
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


def _extract_articles(raw_text: str) -> dict[str, list[dict[str, str]] | dict[str, int | str]]:
    candidates = _extract_article_candidates(raw_text)
    annex_positions = [match.start() for match in ANNEX_RE.finditer(raw_text)]
    line_starts = _line_starts(raw_text)
    resuelve_start = _find_resuelve_start(raw_text)
    used_resuelve_fallback = False

    if resuelve_start is None:
        for candidate in candidates:
            if not _is_hard_negative_header(candidate.header_text):
                resuelve_start = candidate.start_char
                used_resuelve_fallback = True
                break

    articles: list[dict[str, str]] = []
    expected_next = 1
    rewrite_lock = False
    rewrite_lock_line: int | None = None
    rewrite_lock_triggers = 0
    rewrite_lock_timeout_releases = 0
    embedded_skipped = 0
    sequence_jumps = 0
    accepted_count = 0
    last_accepted_base: int | None = None

    for idx, candidate in enumerate(candidates):
        start = candidate.start_char
        next_article = (
            candidates[idx + 1].start_char if idx + 1 < len(candidates) else len(raw_text)
        )
        next_annex = _next_boundary(annex_positions, start)
        end = min(next_article, next_annex) if next_annex is not None else next_article
        slice_text, start_char, end_char = _trim_slice(raw_text, start, end)
        start_line, end_line = _span_lines(raw_text, start_char, end_char, line_starts)

        if _is_hard_negative_header(candidate.header_text):
            embedded_skipped += 1
            continue

        if rewrite_lock:
            release_lock, release_reason = _should_release_rewrite_lock(
                raw_text,
                candidate,
                rewrite_lock_line,
            )
            if release_lock:
                rewrite_lock = False
                rewrite_lock_line = None
                if release_reason == "timeout":
                    rewrite_lock_timeout_releases += 1

        if rewrite_lock:
            embedded_skipped += 1
            continue

        if resuelve_start is None or start < resuelve_start:
            continue

        should_accept = False
        if accepted_count == 0:
            should_accept = candidate.suffix is None
        elif candidate.suffix:
            should_accept = (
                last_accepted_base is not None
                and candidate.number_base == last_accepted_base
            )
        else:
            should_accept = (
                candidate.number_base == expected_next
                or candidate.number_base == expected_next + 1
            )

        if not should_accept:
            if accepted_count > 0 and not candidate.suffix and candidate.number_base > expected_next + 1:
                sequence_jumps += 1
            embedded_skipped += 1
            continue

        articles.append(
            {
                "number": candidate.number_raw,
                "text": slice_text,
                "_range": (start_char, end_char, start_line, end_line),
            }
        )

        accepted_count += 1
        if not candidate.suffix:
            expected_next = candidate.number_base + 1
            last_accepted_base = candidate.number_base

        if _contains_rewrite_trigger(slice_text):
            rewrite_lock = True
            rewrite_lock_line = end_line
            rewrite_lock_triggers += 1

    confidence_label = "ALTA" if articles and sequence_jumps == 0 else "MEDIA" if articles else "BAJA"
    metrics: dict[str, int | str] = {
        "candidates_total": len(candidates),
        "articles_structural_inserted": len(articles),
        "articles_embedded_skipped": embedded_skipped,
        "rewrite_lock_triggers": rewrite_lock_triggers,
        "rewrite_lock_timeout_releases": rewrite_lock_timeout_releases,
        "sequence_jumps_detected": sequence_jumps,
        "used_resuelve_fallback": int(used_resuelve_fallback),
        "structure_confidence": confidence_label,
    }
    return {"articles": articles, "metrics": metrics}


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
    metrics: dict[str, int | str],
) -> float:
    inserted = int(metrics.get("articles_structural_inserted", 0))
    jumps = int(metrics.get("sequence_jumps_detected", 0))
    if inserted == 0:
        return 0.3
    if jumps > 0:
        return 0.65

    score = 0.7
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


def _extract_article_candidates(raw_text: str) -> list[ArticleCandidate]:
    line_starts = _line_starts(raw_text)
    candidates: list[ArticleCandidate] = []
    for match in ARTICLE_RE.finditer(raw_text):
        number_raw = re.sub(r"\s+", " ", match.group(1).strip())
        parsed = _parse_article_token(number_raw)
        if parsed is None:
            continue
        number_base, suffix = parsed
        number_raw = f"{number_base} {suffix}" if suffix else str(number_base)
        start_char = match.start()
        start_line = _line_for_char(start_char, line_starts)
        line_end = raw_text.find("\n", start_char)
        header_line = raw_text[start_char:] if line_end == -1 else raw_text[start_char:line_end]
        candidates.append(
            ArticleCandidate(
                number_raw=number_raw,
                number_base=number_base,
                suffix=suffix,
                header_text=header_line,
                start_char=start_char,
                start_line=start_line,
            )
        )
    return candidates


def _parse_article_token(value: str) -> tuple[int, str | None] | None:
    match = re.match(
        r"^(\d+)(?:\s*(bis|ter|quater|quinquies))?$",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), (match.group(2).lower() if match.group(2) else None)


def _is_hard_negative_header(header_text: str) -> bool:
    return bool(re.match(r'^\s*[-–—]?\s*[“"]', header_text))


def _find_resuelve_start(raw_text: str) -> int | None:
    for match in SECTION_HEADER_RE.finditer(raw_text):
        heading = match.group(1).strip().upper()
        if heading.startswith("RESUEL"):
            return match.start()
    return None


def _contains_rewrite_trigger(text: str) -> bool:
    normalized = _normalize_for_matching(text[:1200])
    triggers = (
        "quedara redactad",
        "de la siguiente manera",
        "sustituyese",
        "incorporase",
        "reemplazase",
        "el que quedara redactado",
    )
    return any(trigger in normalized for trigger in triggers)


def _should_release_rewrite_lock(
    raw_text: str,
    candidate: ArticleCandidate,
    rewrite_lock_line: int | None,
) -> tuple[bool, str | None]:
    if rewrite_lock_line is not None and candidate.start_line - rewrite_lock_line > 45:
        return True, "timeout"

    short_segment = raw_text[max(0, candidate.start_char - 300) : candidate.start_char]
    margin_aligned = len(candidate.header_text) == len(candidate.header_text.lstrip())
    if margin_aligned and not _contains_rewrite_trigger(short_segment):
        return True, "margin_no_trigger"

    quote_count = short_segment.count('"')
    if "”" in short_segment or quote_count >= 2:
        return True, "quotes"
    return False, None


def _normalize_for_matching(text: str) -> str:
    replacements = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    return text.translate(replacements).lower()


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
