import sqlite3
from pathlib import Path

from rg_atp_pipeline.storage.migrations import ensure_schema


def test_relation_llm_reviews_has_raw_item_column(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(relation_llm_reviews)").fetchall()}

    assert "raw_item" in columns
