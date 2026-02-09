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
