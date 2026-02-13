from pathlib import Path

from rg_atp_pipeline.relations_ui import get_relations_summary, get_relations_table
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_relations_summary_and_table(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_schema(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                1,
                "RG-2024-001",
                "LEY-1",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "12",
                "REGEX",
                0.91,
                "snippet 1",
                "exp 1",
                "2025-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2,
                2,
                "RG-2024-002",
                "LEY-2",
                "REPEALS",
                "OUTGOING",
                "NORM",
                None,
                "REGEX",
                0.95,
                "snippet 2",
                "exp 2",
                "2025-01-02T00:00:00Z",
            ),
        )
        conn.commit()

    summary = get_relations_summary(db_path)
    assert summary["total_relations"] == 2
    assert summary["by_type"]["MODIFIES"] == 1
    assert summary["by_type"]["REPEALS"] == 1

    table = get_relations_table(db_path, limit=50)
    expected_cols = {
        "relation_id",
        "source_doc_key",
        "target_norm_key",
        "relation_type",
        "direction",
        "scope",
        "scope_detail",
        "confidence",
        "method",
        "created_at",
        "evidence_snippet",
        "explanation",
    }
    if hasattr(table, "columns"):
        assert expected_cols.issubset(set(table.columns))
        assert len(table) == 2
    else:
        assert len(table) == 2
        assert expected_cols.issubset(set(table[0].keys()))


def test_get_relations_table_uses_relation_extractions_evidence_snippet(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_schema(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                10,
                10,
                "RG-2024-777",
                "LEY-777",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "5",
                "REGEX",
                0.9,
                "Modifícase el artículo 5 de la ley vigente",
                "regex match",
                "2025-01-03T00:00:00Z",
            ),
        )
        conn.commit()

    table = get_relations_table(db_path, limit=10)
    if hasattr(table, "iloc"):
        evidence = str(table.iloc[0]["evidence_snippet"])
    else:
        evidence = str(table[0]["evidence_snippet"])
    assert "Modifícase" in evidence


def test_get_relations_table_prompt_version_keeps_rows_without_review(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_schema(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                101,
                101,
                "RG-2024-101",
                "LEY-101",
                "ACCORDING_TO",
                "OUTGOING",
                "NORM",
                None,
                "REGEX",
                0.88,
                "snippet 101",
                "exp 101",
                "2025-01-10T00:00:00Z",
            ),
        )
        first_relation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                102,
                102,
                "RG-2024-102",
                "LEY-102",
                "ACCORDING_TO",
                "OUTGOING",
                "NORM",
                None,
                "REGEX",
                0.85,
                "snippet 102",
                "exp 102",
                "2025-01-11T00:00:00Z",
            ),
        )
        second_relation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_llm_reviews (
                relation_id, llm_model, prompt_version, relation_type, direction,
                scope, scope_detail, llm_confidence, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_relation_id,
                "gpt-test",
                "reltype-v2",
                "ACCORDING_TO",
                "OUTGOING",
                "NORM",
                None,
                0.93,
                "llm ok",
                "2025-01-12T00:00:00Z",
            ),
        )
        conn.commit()

    table = get_relations_table(db_path, prompt_version="reltype-v2", relation_type="ACCORDING_TO", limit=10)

    if hasattr(table, "to_dict"):
        records = table.to_dict("records")
    else:
        records = table

    assert len(records) == 2

    rows_by_relation_id = {row["relation_id"]: row for row in records}
    assert rows_by_relation_id[first_relation_id]["llm_explanation"] == "llm ok"
    assert rows_by_relation_id[first_relation_id]["prompt_version"] == "reltype-v2"
    assert rows_by_relation_id[second_relation_id]["llm_explanation"] is None
    assert rows_by_relation_id[second_relation_id]["llm_confidence"] is None
    assert rows_by_relation_id[second_relation_id]["llm_model"] is None
    assert rows_by_relation_id[second_relation_id]["prompt_version"] is None
