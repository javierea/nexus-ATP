from __future__ import annotations

import json
import sqlite3

from typer.testing import CliRunner

from rg_atp_pipeline.cli import app


def test_relations_index_diagnose_detects_legacy_index(tmp_path):
    db_path = tmp_path / "rg_atp.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE relation_extractions (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_key TEXT NOT NULL,
                source_unit_id INTEGER,
                target_norm_key TEXT,
                relation_type TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                method TEXT NOT NULL,
                extract_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX ux_relation_extractions_unit_target_type
            ON relation_extractions(
                source_doc_key,
                source_unit_id,
                target_norm_key,
                relation_type
            )
            """
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                source_doc_key, source_unit_id, target_norm_key, relation_type,
                scope, scope_detail, method, extract_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RG-1", 10, None, "REPEALS", "WHOLE_NORM", None, "REGEX", "relext-v2"),
        )
        conn.commit()

    runner = CliRunner()
    result = runner.invoke(app, ["relations-index-diagnose", "--db-path", str(db_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["index_present"] is True
    assert payload["index_matches_expected"] is False
    assert payload["null_target_norm_key_rows"] == 1
    assert payload["null_scope_detail_rows"] == 1
