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
