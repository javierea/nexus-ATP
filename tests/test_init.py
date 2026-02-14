from pathlib import Path
import sqlite3

from rg_atp_pipeline.cli import init_project, validate_config_state
from rg_atp_pipeline.config import default_config, save_config
from rg_atp_pipeline.paths import config_path, data_dir, state_path
from rg_atp_pipeline.state import default_state, save_state
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_init_creates_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))

    init_project()

    assert config_path().exists()
    assert state_path().exists()
    assert (data_dir() / "logs").exists()
    assert (data_dir() / "state" / "rg_atp.sqlite").exists()


def test_validate_config_ok(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    state_path().parent.mkdir(parents=True, exist_ok=True)

    save_config(default_config(), config_path())
    save_state(default_state(), state_path())

    config, state = validate_config_state(config_path(), state_path())
    assert config.user_agent == "rg_atp_pipeline/0.1"
    assert state.schema_version == 1


def test_ensure_schema_skips_orphan_relation_extractions(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            CREATE TABLE citations (
                citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_key TEXT NOT NULL,
                source_unit_id TEXT,
                source_unit_type TEXT,
                raw_text TEXT NOT NULL,
                norm_type_guess TEXT NOT NULL,
                norm_key_candidate TEXT,
                evidence_snippet TEXT NOT NULL,
                regex_confidence REAL NOT NULL,
                detected_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE citation_links (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_id INTEGER NOT NULL,
                target_norm_id INTEGER,
                target_norm_key TEXT,
                resolution_status TEXT NOT NULL,
                resolution_confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE relation_extractions (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_id INTEGER NOT NULL,
                link_id INTEGER,
                source_doc_key TEXT NOT NULL,
                target_norm_key TEXT,
                relation_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                method TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_snippet TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (999, NULL, 'RG-X', 'LEY-1-A', 'MODIFIES', 'SOURCE_TO_TARGET',
                      'WHOLE_NORM', NULL, 'REGEX', 0.8, 'snippet', 'orphan row',
                      '2026-01-01T00:00:00Z')
            """
        )
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
    assert count == 0
