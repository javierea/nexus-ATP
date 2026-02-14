from pathlib import Path

from rg_atp_pipeline.queries import (
    get_structure_anomalies,
    get_structure_summary,
    get_units_for_doc,
)
from rg_atp_pipeline.storage_sqlite import DocumentStore


def test_structure_summary_anomalies_and_units(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = DocumentStore(db_path)
    store.initialize()

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                doc_key, url, doc_family, year, number,
                first_seen_at, last_checked_at, status, text_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-001",
                "http://example/1",
                "RG",
                2024,
                1,
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "DOWNLOADED",
                "EXTRACTED",
            ),
        )
        conn.execute(
            """
            INSERT INTO documents (
                doc_key, url, doc_family, year, number,
                first_seen_at, last_checked_at, status, text_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-002",
                "http://example/2",
                "RG",
                2024,
                2,
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "DOWNLOADED",
                "EXTRACTED",
            ),
        )
        conn.execute(
            """
            INSERT INTO doc_structure (
                doc_key, structure_status, structure_confidence,
                articles_detected, annexes_detected, notes, structured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-001",
                "STRUCTURED",
                0.92,
                5,
                0,
                "{}",
                "2025-01-02T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO doc_structure (
                doc_key, structure_status, structure_confidence,
                articles_detected, annexes_detected, notes, structured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-002",
                "PARTIAL",
                0.45,
                0,
                0,
                "{}",
                "2025-01-03T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO units (
                doc_key, unit_type, unit_number, title, text,
                start_char, end_char, start_line, end_line, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-001",
                "ARTICULO",
                "1",
                None,
                "Artículo 1 texto",
                0,
                15,
                1,
                1,
                "2025-01-02T00:00:00Z",
            ),
        )
        conn.commit()

    summary = get_structure_summary(db_path)
    assert summary["total_docs"] == 2
    assert summary["structured_ok"] == 1
    assert summary["partial"] == 1
    assert summary["units_total"] == 1

    anomalies = get_structure_anomalies(db_path, {"confidence_lt": 0.6})
    assert len(anomalies) == 1
    assert anomalies[0]["doc_key"] == "RG-2024-002"

    units = get_units_for_doc(db_path, "RG-2024-001")
    assert len(units) == 1
    assert units[0]["unit_type"] == "ARTICULO"
