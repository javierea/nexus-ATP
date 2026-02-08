import logging
from pathlib import Path

from reportlab.pdfgen import canvas

from rg_atp_pipeline.config import Config, TextQualityConfig, default_config
from rg_atp_pipeline.storage_sqlite import DocumentRecord, DocumentStore
from rg_atp_pipeline.text_extractor import ExtractOptions, run_extract


def _make_pdf(path: Path, text: str | None = None) -> None:
    pdf = canvas.Canvas(str(path))
    if text:
        pdf.drawString(100, 750, text)
    pdf.showPage()
    pdf.save()


def _record_for_pdf(doc_key: str, pdf_path: Path) -> DocumentRecord:
    return DocumentRecord(
        doc_key=doc_key,
        url="https://example.com/doc.pdf",
        doc_family="TEST",
        year=2024,
        number=1,
        first_seen_at="2025-01-01T00:00:00+00:00",
        last_checked_at="2025-01-01T00:00:00+00:00",
        last_downloaded_at="2025-01-01T00:00:00+00:00",
        latest_sha256=None,
        latest_pdf_path=str(pdf_path),
        status="DOWNLOADED",
        http_status=200,
        error_message=None,
        text_status="NONE",
        text_path=None,
        text_extracted_at=None,
        char_count=None,
        pages_total=None,
        pages_with_text=None,
        alpha_ratio=None,
    )


def _config_with_quality() -> Config:
    config = default_config()
    return config.model_copy(
        update={
            "text_quality": TextQualityConfig(
                min_chars_total=1,
                min_chars_per_page=1,
                min_alpha_ratio=0.0,
            )
        }
    )


def test_extract_text_success(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf_path = tmp_path / "with_text.pdf"
    _make_pdf(pdf_path, text="Hola ATP")

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_pdf("DOC-1", pdf_path))

    summary = run_extract(
        _config_with_quality(),
        store,
        data_dir,
        ExtractOptions(
            status="DOWNLOADED",
            limit=None,
            doc_key=None,
            force=False,
            only_text=False,
            only_needs_ocr=False,
        ),
        logger=logging.getLogger(__name__),
    )

    assert summary.extracted == 1
    record = store.get_document("DOC-1")
    assert record is not None
    assert record.text_status == "EXTRACTED"
    assert record.char_count and record.char_count > 0
    assert record.pages_with_text and record.pages_with_text >= 1
    assert record.text_path and Path(record.text_path).exists()


def test_extract_needs_ocr_when_empty(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf_path = tmp_path / "empty.pdf"
    _make_pdf(pdf_path, text=None)

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_pdf("DOC-2", pdf_path))

    config = default_config().model_copy(
        update={
            "text_quality": TextQualityConfig(
                min_chars_total=10,
                min_chars_per_page=5,
                min_alpha_ratio=0.5,
            )
        }
    )

    summary = run_extract(
        config,
        store,
        data_dir,
        ExtractOptions(
            status="DOWNLOADED",
            limit=None,
            doc_key=None,
            force=False,
            only_text=False,
            only_needs_ocr=False,
        ),
        logger=logging.getLogger(__name__),
    )

    assert summary.needs_ocr == 1
    record = store.get_document("DOC-2")
    assert record is not None
    assert record.text_status == "NEEDS_OCR"
    assert record.pages_with_text == 0


def test_extract_idempotent_without_force(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf_path = tmp_path / "with_text.pdf"
    _make_pdf(pdf_path, text="Texto breve")

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_pdf("DOC-3", pdf_path))

    config = _config_with_quality()
    options = ExtractOptions(
        status="DOWNLOADED",
        limit=None,
        doc_key=None,
        force=False,
        only_text=False,
        only_needs_ocr=False,
    )

    run_extract(config, store, data_dir, options, logger=logging.getLogger(__name__))
    record_first = store.get_document("DOC-3")
    assert record_first is not None

    run_extract(config, store, data_dir, options, logger=logging.getLogger(__name__))
    record_second = store.get_document("DOC-3")
    assert record_second is not None
    assert record_second.text_extracted_at == record_first.text_extracted_at
    assert record_second.char_count == record_first.char_count
