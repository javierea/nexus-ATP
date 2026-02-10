import sqlite3
from pathlib import Path

from rg_atp_pipeline.services.citations_service import run_citations
from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def test_citations_service_resolves_and_creates_placeholders(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-001"
    raw_text = (
        "Conforme Ley 83-F y Dec. Ley 2444/62, "
        "además de la Ley 9999-Z."
    )
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    norm_id = repo.upsert_norm("LEY-83-F", "LEY")
    repo.add_alias(norm_id, "Dec. Ley 2444/62", alias_kind="CITATION", confidence=0.9)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert summary["citations_inserted"] >= 2

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        links = conn.execute(
            "SELECT resolution_status, target_norm_key FROM citation_links"
        ).fetchall()
        statuses = {row["resolution_status"] for row in links}
        assert "RESOLVED" in statuses
        assert "PLACEHOLDER_CREATED" in statuses
        placeholder = conn.execute(
            "SELECT norm_key FROM norms WHERE norm_key = 'LEY-9999-Z'"
        ).fetchone()
        assert placeholder is not None


def test_citations_service_dedupes_duplicate_candidates(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-002"
    raw_text = "Ley 83-F. Ley 83-F."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=False,
    )

    assert summary["citations_inserted"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        citations_count = conn.execute(
            "SELECT COUNT(*) AS total FROM citations"
        ).fetchone()["total"]
        links_count = conn.execute(
            "SELECT COUNT(*) AS total FROM citation_links"
        ).fetchone()["total"]
        assert citations_count == 1
        assert links_count == 1


def test_citations_service_placeholder_fk(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-003"
    raw_text = "Según Ley 1234-A."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert summary["placeholders_created"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT l.target_norm_id, l.target_norm_key, n.norm_key
            FROM citation_links l
            JOIN norms n ON n.norm_id = l.target_norm_id
            """
        ).fetchone()
        assert row is not None
        assert row["target_norm_key"] == "LEY-1234-A"
        assert row["norm_key"] == "LEY-1234-A"


def test_citations_service_runs_twice_without_lock(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-004"
    raw_text = "Ley 83-F y Dec. Ley 2444/62."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    norm_id = repo.upsert_norm("LEY-83-F", "LEY")
    repo.add_alias(norm_id, "Dec. Ley 2444/62", alias_kind="CITATION", confidence=0.9)

    summaries = []
    for _ in range(2):
        summaries.append(
            run_citations(
                db_path=db_path,
                data_dir=data_dir,
                doc_keys=[doc_key],
                limit_docs=None,
                llm_mode="off",
                min_confidence=0.7,
                create_placeholders=True,
            )
        )

    assert summaries[0]["citations_inserted"] >= 1


def test_citations_service_streamlit_rerun(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-005"
    raw_text = "Según Ley 1234-A."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    first = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )
    second = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert first["docs_processed"] == 1
    assert second["docs_processed"] == 1


def test_citations_summary_matches_db_counts(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-006"
    raw_text = "Ley 83-F y Ley 9999-Z."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    repo.upsert_norm("LEY-83-F", "LEY")

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT resolution_status, COUNT(*) AS total
            FROM citation_links
            GROUP BY resolution_status
            """
        ).fetchall()
        counts = {row["resolution_status"]: row["total"] for row in rows}

    assert summary["resolved"] == counts.get("RESOLVED", 0)
    assert summary["placeholders_created"] == counts.get("PLACEHOLDER_CREATED", 0)
    assert summary["unresolved"] == counts.get("UNRESOLVED", 0)
    assert summary["rejected"] == counts.get("REJECTED", 0)
