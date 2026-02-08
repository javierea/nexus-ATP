"""Text extraction for rg_atp_pipeline (Etapa 2)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from .config import Config
from .storage_sqlite import DocumentStore


@dataclass(frozen=True)
class TextMetrics:
    char_count: int
    pages_total: int
    pages_with_text: int
    alpha_ratio: float


@dataclass(frozen=True)
class ExtractOptions:
    status: str
    limit: int | None
    doc_key: str | None
    force: bool
    only_text: bool
    only_needs_ocr: bool


@dataclass(frozen=True)
class ExtractSummary:
    processed: int
    extracted: int
    needs_ocr: int
    error: int

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "extracted": self.extracted,
            "needs_ocr": self.needs_ocr,
            "error": self.error,
        }


def run_extract(
    config: Config,
    store: DocumentStore,
    data_dir: Path,
    options: ExtractOptions,
    logger: logging.Logger,
) -> ExtractSummary:
    store.initialize()
    quality = config.text_quality

    if options.only_text and options.only_needs_ocr:
        raise ValueError("No se puede combinar --only-text y --only-needs-ocr.")

    only_status = None
    if options.only_text:
        only_status = "EXTRACTED"
    elif options.only_needs_ocr:
        only_status = "NEEDS_OCR"

    candidates = store.list_text_candidates(
        status=options.status,
        limit=options.limit,
        doc_key=options.doc_key,
        only_text_status=only_status,
    )

    processed = 0
    extracted = 0
    needs_ocr = 0
    error = 0

    text_dir = data_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    for record in candidates:
        if not record.latest_pdf_path:
            logger.warning("Sin PDF para %s (status=%s).", record.doc_key, record.status)
            store.update_text_info(
                record.doc_key,
                "ERROR",
                None,
                _now_iso(),
                None,
                None,
                None,
                None,
            )
            error += 1
            processed += 1
            continue

        if not options.force and record.text_status in {"EXTRACTED", "NEEDS_OCR"}:
            logger.info("Omitido %s (text_status=%s).", record.doc_key, record.text_status)
            continue

        processed += 1
        pdf_path = Path(record.latest_pdf_path)
        if not pdf_path.exists():
            logger.warning("PDF no encontrado %s para %s.", pdf_path, record.doc_key)
            store.update_text_info(
                record.doc_key,
                "ERROR",
                None,
                _now_iso(),
                None,
                None,
                None,
                None,
            )
            error += 1
            continue

        try:
            content, metrics = _extract_text(pdf_path, quality.min_chars_per_page)
            text_path = text_dir / f"{record.doc_key}.txt"
            text_path.write_text(content, encoding="utf-8")
            needs_ocr_flag = _needs_ocr(metrics, quality.min_chars_total, quality.min_alpha_ratio)
            text_status = "NEEDS_OCR" if needs_ocr_flag else "EXTRACTED"
            store.update_text_info(
                record.doc_key,
                text_status,
                str(text_path),
                _now_iso(),
                metrics.char_count,
                metrics.pages_total,
                metrics.pages_with_text,
                metrics.alpha_ratio,
            )
            if needs_ocr_flag:
                needs_ocr += 1
                logger.info("NEEDS_OCR %s (%s chars).", record.doc_key, metrics.char_count)
            else:
                extracted += 1
                logger.info("EXTRACTED %s (%s chars).", record.doc_key, metrics.char_count)
        except Exception as exc:  # noqa: BLE001 - keep pipeline running
            logger.exception("ERROR extracción %s: %s", record.doc_key, exc)
            store.update_text_info(
                record.doc_key,
                "ERROR",
                None,
                _now_iso(),
                None,
                None,
                None,
                None,
            )
            error += 1

    return ExtractSummary(
        processed=processed,
        extracted=extracted,
        needs_ocr=needs_ocr,
        error=error,
    )


def _extract_text(pdf_path: Path, min_chars_per_page: int) -> tuple[str, TextMetrics]:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    raw_page_texts: list[str] = []
    pages_total = len(reader.pages)
    pages_with_text = 0

    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) >= min_chars_per_page:
            pages_with_text += 1
        page_texts.append(f"===PAGE {idx}===\n{page_text}\n")
        raw_page_texts.append(page_text)

    raw_text = "".join(page_texts)
    metrics = _compute_metrics("".join(raw_page_texts), pages_total, pages_with_text)
    return raw_text, metrics


def _compute_metrics(text: str, pages_total: int, pages_with_text: int) -> TextMetrics:
    char_count = len(text)
    non_space = re.sub(r"\s+", "", text)
    alnum = sum(1 for ch in non_space if ch.isalnum())
    alpha_ratio = alnum / len(non_space) if non_space else 0.0
    return TextMetrics(
        char_count=char_count,
        pages_total=pages_total,
        pages_with_text=pages_with_text,
        alpha_ratio=alpha_ratio,
    )


def _needs_ocr(metrics: TextMetrics, min_chars_total: int, min_alpha_ratio: float) -> bool:
    if metrics.char_count < min_chars_total:
        return True
    if metrics.pages_with_text == 0:
        return True
    if metrics.alpha_ratio < min_alpha_ratio:
        return True
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
