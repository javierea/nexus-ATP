import sqlite3
import json
from pathlib import Path

from rg_atp_pipeline.services.relations_service import run_relations
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_relations_service_inserts_regex_relations(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-900",
                "1",
                "ARTICLE",
                "Derógase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase la Ley 83-F de forma integral",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'PLACEHOLDER_CREATED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-900"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 1
    assert summary["by_type"].get("REPEALS") == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT relation_type, scope, method FROM relation_extractions"
        ).fetchone()
    assert row is not None
    assert row["relation_type"] == "REPEALS"
    assert row["scope"] == "WHOLE_NORM"
    assert row["method"] == "REGEX"


def test_relations_service_uses_structured_unit_text(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    structured_dir = data_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    doc_key = "RG-2024-901"
    unit_id = "U-1"
    structured_payload = {
        "doc_key": doc_key,
        "units": [
            {
                "unit_id": unit_id,
                "unit_type": "ARTICLE",
                "text": "Texto inicial. Derógase la Ley 83-F. Texto final.",
            }
        ],
    }
    (structured_dir / f"{doc_key}.json").write_text(
        json.dumps(structured_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_key,
                unit_id,
                "ARTICLE",
                "Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Ley 83-F",
                0.7,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 1
    assert summary["by_type"].get("REPEALS") == 1


def test_relations_service_ignores_unresolved_links(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-902",
                "1",
                "ARTICLE",
                "Derógase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase la Ley 83-F",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'UNRESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.2, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-902"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["links_seen"] == 0
    assert summary["relations_inserted"] == 0
