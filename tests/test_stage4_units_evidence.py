import sqlite3
from pathlib import Path

from rg_atp_pipeline.queries import find_unit_for_offset
from rg_atp_pipeline.services.citations_service import run_citations
from rg_atp_pipeline.services.relations_service import run_relations
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_find_unit_for_offset_prefers_articulo(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO units (doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("RG-1", "CONSIDERANDO", None, None, "A" * 300, 0, 300, 1, 10, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO units (doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("RG-1", "ARTICULO", "1", None, "Artículo 1", 10, 120, 2, 4, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    assert find_unit_for_offset(db_path, "RG-1", 20, 30) == 2


def test_citations_persist_unit_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "structured").mkdir(parents=True)
    (data_dir / "state").mkdir(parents=True)
    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    text = "Artículo 1: Modifícase la Ley 83-F."
    (data_dir / "structured" / "RG-2.json").write_text(
        '{"units":[{"unit_id":"1","unit_type":"ARTICULO","unit_number":"1","start_char":0,"end_char":35,"text":"'
        + text
        + '"}]}',
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-2", "ARTICULO", "1", None, text, 0, len(text), 1, 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=["RG-2"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.5,
        create_placeholders=False,
        extract_version="citext-v2",
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT evidence_kind, evidence_unit_id, LENGTH(evidence_text) FROM citations LIMIT 1"
        ).fetchone()
    assert row[0] == "UNIT"
    assert row[1] == 1
    assert row[2] <= 520


def test_citations_coexist_by_extract_version(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "structured").mkdir(parents=True)
    (data_dir / "state").mkdir(parents=True)
    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    text = "Artículo 1: Modifícase la Ley 83-F."
    (data_dir / "structured" / "RG-2B.json").write_text(
        '{"units":[{"unit_id":"1","unit_type":"ARTICULO","unit_number":"1","start_char":0,"end_char":35,"text":"'
        + text
        + '"}]}',
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-2B", "ARTICULO", "1", None, text, 0, len(text), 1, 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=["RG-2B"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.5,
        create_placeholders=False,
        extract_version="citext-v2",
    )
    run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=["RG-2B"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.5,
        create_placeholders=False,
        extract_version="citext-v3",
    )
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    assert count == 2


def test_relations_store_source_unit_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO citations (source_doc_key, source_unit_id, source_unit_type, raw_text, norm_type_guess, norm_key_candidate, evidence_snippet, evidence_unit_id, evidence_text, evidence_kind, regex_confidence, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "RG-3",
                "1",
                "ARTICULO",
                "Ley 83-F",
                "LEY",
                "LEY-83-F",
                "snippet",
                1,
                "Artículo 1: Derógase la Ley 83-F.",
                "UNIT",
                0.9,
                "2026-01-01T00:00:00Z",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO citation_links (citation_id, target_norm_id, target_norm_key, resolution_status, resolution_confidence, created_at) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)",
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-3", "ARTICULO", "1", None, "Artículo 1: Derógase la Ley 83-F.", 0, 33, 1, 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-3"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v3",
        batch_size=5,
        extract_version="relext-v2",
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT source_unit_id, source_unit_number, extract_version FROM relation_extractions LIMIT 1"
        ).fetchone()
    assert row == (1, "1", "relext-v2")


def test_relations_coexist_by_extract_version(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO citations (source_doc_key, source_unit_id, source_unit_type, raw_text, norm_type_guess, norm_key_candidate, evidence_snippet, evidence_unit_id, evidence_text, evidence_kind, regex_confidence, detected_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "RG-4",
                "1",
                "ARTICULO",
                "Ley 83-F",
                "LEY",
                "LEY-83-F",
                "snippet",
                1,
                "Artículo 1: Derógase la Ley 83-F.",
                "UNIT",
                0.9,
                "2026-01-01T00:00:00Z",
                "citext-v2",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO citation_links (citation_id, target_norm_id, target_norm_key, resolution_status, resolution_confidence, created_at) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)",
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-4", "ARTICULO", "1", None, "Artículo 1: Derógase la Ley 83-F.", 0, 33, 1, 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    run_relations(db_path=db_path, data_dir=tmp_path, doc_keys=["RG-4"], limit_docs=None, llm_mode="off", min_confidence=0.6, prompt_version="reltype-v3", batch_size=5, extract_version="relext-v2")
    run_relations(db_path=db_path, data_dir=tmp_path, doc_keys=["RG-4"], limit_docs=None, llm_mode="off", min_confidence=0.6, prompt_version="reltype-v3", batch_size=5, extract_version="relext-v3")
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
    assert count == 2


def test_relations_filter_by_citation_extract_version(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        for idx, cit_ver in enumerate(["citext-v1", "citext-v2"], start=1):
            conn.execute(
                "INSERT INTO citations (source_doc_key, source_unit_id, source_unit_type, raw_text, norm_type_guess, norm_key_candidate, evidence_snippet, evidence_unit_id, evidence_text, evidence_kind, regex_confidence, detected_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "RG-5",
                    str(idx),
                    "ARTICULO",
                    "Ley 83-F",
                    "LEY",
                    "LEY-83-F",
                    "snippet",
                    idx,
                    f"Artículo {idx}: Derógase la Ley 83-F.",
                    "UNIT",
                    0.9,
                    f"2026-01-0{idx}T00:00:00Z",
                    cit_ver,
                ),
            )
            citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO citation_links (citation_id, target_norm_id, target_norm_key, resolution_status, resolution_confidence, created_at) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)",
                (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00Z"),
            )
            conn.execute(
                "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (idx, "RG-5", "ARTICULO", str(idx), None, f"Artículo {idx}: Derógase la Ley 83-F.", 0, 33, 1, 1, "2026-01-01T00:00:00Z"),
            )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-5"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v3",
        batch_size=5,
        extract_version="relext-v2",
        citation_extract_version="citext-v2",
    )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
    assert summary["citation_extract_version_effective"] == "citext-v2"
    assert count == 1


def test_relations_materialize_intra_norm_edges(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-6", "ARTICULO", "1", None, "Artículo 1: Ver artículo 2.", 0, 30, 1, 1, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO units (id, doc_key, unit_type, unit_number, title, text, start_char, end_char, start_line, end_line, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2, "RG-6", "ARTICULO", "2", None, "Artículo 2: Contenido.", 31, 60, 2, 2, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO citations (source_doc_key, source_unit_id, source_unit_type, raw_text, norm_type_guess, norm_key_candidate, evidence_snippet, evidence_unit_id, evidence_text, evidence_kind, regex_confidence, detected_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "RG-6",
                "1",
                "ARTICULO",
                "Ley 83-F",
                "LEY",
                "LEY-83-F",
                "snippet",
                1,
                "Artículo 1: Derógase la Ley 83-F. Ver artículo 2.",
                "UNIT",
                0.9,
                "2026-01-01T00:00:00Z",
                "citext-v2",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO citation_links (citation_id, target_norm_id, target_norm_key, resolution_status, resolution_confidence, created_at) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)",
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-6"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v3",
        batch_size=5,
        extract_version="relext-v2",
    )

    with sqlite3.connect(db_path) as conn:
        edge = conn.execute(
            "SELECT source_unit_id, target_unit_id, relation_type FROM intra_norm_relations"
        ).fetchone()
    assert summary["intra_norm_relations_inserted"] == 1
    assert edge == (1, 2, "REFERS_INTERNAL_ARTICLE")
