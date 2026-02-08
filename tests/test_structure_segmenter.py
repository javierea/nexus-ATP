import logging
from pathlib import Path

from rg_atp_pipeline.storage_sqlite import DocumentRecord, DocumentStore
from rg_atp_pipeline.structure_segmenter import StructureOptions, run_structure


def _record_for_text(doc_key: str, text_path: Path) -> DocumentRecord:
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
        latest_pdf_path=str(text_path),
        status="DOWNLOADED",
        http_status=200,
        error_message=None,
        text_status="EXTRACTED",
        text_path=str(text_path),
        text_extracted_at="2025-01-01T00:00:00+00:00",
        char_count=len(text_path.read_text(encoding="utf-8")),
        pages_total=1,
        pages_with_text=1,
        alpha_ratio=0.9,
    )


def test_structure_detects_articles(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    text_dir = data_dir / "text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "DOC-1.txt"
    text_path.write_text(
        "VISTO:\nAlgo\nCONSIDERANDO:\nMás texto\nRESUELVE:\n"
        "ARTÍCULO 1°.- Primero.\nARTÍCULO 2°.- Segundo.\n",
        encoding="utf-8",
    )

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_text("DOC-1", text_path))

    summary = run_structure(
        store,
        data_dir,
        StructureOptions(
            doc_key=None,
            limit=None,
            force=False,
            include_needs_ocr=False,
            export_json=False,
        ),
        logger=logging.getLogger(__name__),
    )

    assert summary.structured_ok == 1
    doc_structure = store.get_doc_structure("DOC-1")
    assert doc_structure is not None
    assert doc_structure.structure_status == "STRUCTURED"
    assert doc_structure.articles_detected == 2
    assert doc_structure.structure_confidence and doc_structure.structure_confidence > 0.6
    units = store.list_units("DOC-1")
    article_units = [unit for unit in units if unit.unit_type == "ARTICULO"]
    assert len(article_units) == 2


def test_structure_without_articles_is_partial(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    text_dir = data_dir / "text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "DOC-2.txt"
    text_path.write_text("Texto sin artículos.\nSolo párrafos.", encoding="utf-8")

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_text("DOC-2", text_path))

    run_structure(
        store,
        data_dir,
        StructureOptions(
            doc_key=None,
            limit=None,
            force=False,
            include_needs_ocr=False,
            export_json=False,
        ),
        logger=logging.getLogger(__name__),
    )

    doc_structure = store.get_doc_structure("DOC-2")
    assert doc_structure is not None
    assert doc_structure.structure_status == "PARTIAL"
    units = store.list_units("DOC-2")
    assert any(unit.unit_type == "OTRO" for unit in units)


def test_structure_detects_annex(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    text_dir = data_dir / "text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "DOC-3.txt"
    text_path.write_text(
        "RESUELVE:\nARTÍCULO 1°.- Texto.\n\nANEXO I\nDetalle del anexo.\n",
        encoding="utf-8",
    )

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_text("DOC-3", text_path))

    run_structure(
        store,
        data_dir,
        StructureOptions(
            doc_key=None,
            limit=None,
            force=False,
            include_needs_ocr=False,
            export_json=False,
        ),
        logger=logging.getLogger(__name__),
    )

    units = store.list_units("DOC-3")
    assert any(unit.unit_type == "ANEXO" for unit in units)


def test_structure_idempotent_without_force(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    text_dir = data_dir / "text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "DOC-4.txt"
    text_path.write_text(
        "RESUELVE:\nARTÍCULO 1°.- Texto.\nARTÍCULO 2°.- Texto.\n",
        encoding="utf-8",
    )

    store = DocumentStore(data_dir / "state" / "rg_atp.sqlite")
    store.initialize()
    store.upsert(_record_for_text("DOC-4", text_path))

    options = StructureOptions(
        doc_key=None,
        limit=None,
        force=False,
        include_needs_ocr=False,
        export_json=False,
    )

    first = run_structure(store, data_dir, options, logger=logging.getLogger(__name__))
    second = run_structure(store, data_dir, options, logger=logging.getLogger(__name__))

    assert first.structured_ok == 1
    assert second.skipped == 1
