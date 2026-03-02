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


def test_ensure_schema_migrates_relations_with_reviews(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
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
                extract_version TEXT NOT NULL DEFAULT 'citext-v2',
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
                created_at TEXT NOT NULL,
                FOREIGN KEY(citation_id) REFERENCES citations(citation_id)
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
                created_at TEXT NOT NULL,
                FOREIGN KEY(citation_id) REFERENCES citations(citation_id),
                FOREIGN KEY(link_id) REFERENCES citation_links(link_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE relation_llm_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_id INTEGER NOT NULL,
                llm_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                llm_confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(relation_id, llm_model, prompt_version),
                FOREIGN KEY(relation_id) REFERENCES relation_extractions(relation_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO citations (
                citation_id, source_doc_key, source_unit_id, source_unit_type,
                raw_text, norm_type_guess, norm_key_candidate, evidence_snippet,
                extract_version, regex_confidence, detected_at
            ) VALUES (1, 'RG-1', 'u1', 'ART', 'Ley 1', 'LEY', 'LEY-1', 'snippet', 'citext-v2', 0.9, '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                relation_id, citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail, method,
                confidence, evidence_snippet, explanation, created_at
            ) VALUES (
                1, 1, NULL, 'RG-1', 'LEY-2', 'MODIFIES', 'SOURCE_TO_TARGET',
                'WHOLE_NORM', NULL, 'REGEX', 0.8, 'snippet', 'ok', '2026-01-01T00:00:00Z'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO relation_llm_reviews (
                relation_id, llm_model, prompt_version, relation_type, direction,
                scope, scope_detail, llm_confidence, explanation, created_at
            ) VALUES (
                1, 'test-model', 'v1', 'MODIFIES', 'SOURCE_TO_TARGET',
                'WHOLE_NORM', NULL, 0.8, 'ok', '2026-01-01T00:00:00Z'
            )
            """
        )
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(relation_extractions)")}
        relation_count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
        review_count = conn.execute("SELECT COUNT(*) FROM relation_llm_reviews").fetchone()[0]

    assert "extract_version" in cols
    assert relation_count == 1
    assert review_count == 1


def test_ensure_schema_skips_orphan_relation_reviews_on_migration(tmp_path: Path) -> None:
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
                extract_version TEXT NOT NULL DEFAULT 'citext-v2',
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
            CREATE TABLE relation_llm_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_id INTEGER NOT NULL,
                llm_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                llm_confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(relation_id, llm_model, prompt_version),
                FOREIGN KEY(relation_id) REFERENCES relation_extractions(relation_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO citations (
                citation_id, source_doc_key, source_unit_id, source_unit_type,
                raw_text, norm_type_guess, norm_key_candidate, evidence_snippet,
                extract_version, regex_confidence, detected_at
            ) VALUES (1, 'RG-1', 'u1', 'ART', 'Ley 1', 'LEY', 'LEY-1', 'snippet', 'citext-v2', 0.9, '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                relation_id, citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail, method,
                confidence, evidence_snippet, explanation, created_at
            ) VALUES
                (1, 1, NULL, 'RG-1', 'LEY-2', 'MODIFIES', 'SOURCE_TO_TARGET',
                 'WHOLE_NORM', NULL, 'REGEX', 0.8, 'snippet', 'ok', '2026-01-01T00:00:00Z'),
                (2, 999, NULL, 'RG-X', 'LEY-X', 'UNKNOWN', 'UNKNOWN',
                 'UNKNOWN', NULL, 'REGEX', 0.2, 'snippet', 'orphan', '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO relation_llm_reviews (
                relation_id, llm_model, prompt_version, relation_type, direction,
                scope, scope_detail, llm_confidence, explanation, created_at
            ) VALUES
                (1, 'test-model', 'v1', 'MODIFIES', 'SOURCE_TO_TARGET',
                 'WHOLE_NORM', NULL, 0.8, 'ok', '2026-01-01T00:00:00Z'),
                (2, 'test-model', 'v1', 'UNKNOWN', 'UNKNOWN',
                 'UNKNOWN', NULL, 0.2, 'orphan', '2026-01-01T00:00:00Z')
            """
        )
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rel_ids = [
            row[0]
            for row in conn.execute(
                "SELECT relation_id FROM relation_extractions ORDER BY relation_id"
            )
        ]
        review_ids = [
            row[0]
            for row in conn.execute(
                "SELECT relation_id FROM relation_llm_reviews ORDER BY relation_id"
            )
        ]

    assert rel_ids == [1]
    assert review_ids == [1]


def test_ensure_schema_backfills_relation_nullable_unique_fields(tmp_path: Path) -> None:
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
                "RG-2",
                "1",
                "ARTICLE",
                "texto",
                "LEY",
                "LEY-2",
                "snippet",
                0.9,
                "2026-01-01T00:00:00Z",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id,
                "RG-2",
                1,
                "1",
                "texto",
                None,
                "relext-v2",
                "REPEALS",
                "OUTGOING",
                "WHOLE_NORM",
                None,
                "REGEX",
                0.5,
                "snippet",
                "snippet",
                "exp",
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT target_norm_key, scope_detail FROM relation_extractions LIMIT 1"
        ).fetchone()

    assert row is not None
    assert row[0] == ""
    assert row[1] == ""


def test_ensure_schema_dedupes_relation_extractions_before_unique_index(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX IF EXISTS ux_relation_extractions_unit_target_type")
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RG-DUP", "1", "ARTICLE", "texto a", "LEY", "LEY-A", "s1", 0.9, "2026-01-01T00:00:00Z"),
        )
        citation_id_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RG-DUP", "1", "ARTICLE", "texto b", "LEY", "LEY-A", "s2", 0.9, "2026-01-01T00:00:01Z"),
        )
        citation_id_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_1,
                "RG-DUP",
                1,
                "1",
                "texto",
                "LEY-A",
                "relext-v2",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.65,
                "snippet-1",
                "snippet-1",
                "exp-1",
                "2026-01-01T00:00:00Z",
            ),
        )
        relation_id_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_2,
                "RG-DUP",
                1,
                "1",
                "texto",
                "LEY-A",
                "relext-v2",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.91,
                "snippet-2",
                "snippet-2",
                "exp-2",
                "2026-01-01T00:00:02Z",
            ),
        )
        relation_id_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_llm_reviews (
                relation_id, llm_model, prompt_version, relation_type, direction,
                scope, scope_detail, llm_confidence, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (relation_id_1, "model", "v1", "MODIFIES", "OUTGOING", "ARTICLE", "1", 0.7, "low", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO relation_llm_reviews (
                relation_id, llm_model, prompt_version, relation_type, direction,
                scope, scope_detail, llm_confidence, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (relation_id_2, "model", "v1", "MODIFIES", "OUTGOING", "ARTICLE", "1", 0.9, "high", "2026-01-01T00:00:02Z"),
        )
        conn.execute("DROP INDEX IF EXISTS ux_relation_extractions_unit_target_type")
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT relation_id, confidence FROM relation_extractions").fetchall()
        reviews = conn.execute("SELECT relation_id FROM relation_llm_reviews ORDER BY relation_id").fetchall()

    assert rows == [(relation_id_2, 0.91)]
    assert reviews == [(relation_id_2,)]


def test_ensure_schema_handles_null_normalization_with_legacy_unit_unique_index(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX IF EXISTS ux_relation_extractions_unit_target_type")
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
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RG-LEGACY", "10", "ARTICLE", "texto a", "LEY", "LEY-X", "s1", 0.9, "2026-01-01T00:00:00Z"),
        )
        citation_id_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RG-LEGACY", "10", "ARTICLE", "texto b", "LEY", "LEY-X", "s2", 0.9, "2026-01-01T00:00:01Z"),
        )
        citation_id_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_1,
                "RG-LEGACY",
                10,
                "10",
                "texto",
                None,
                "relext-v2",
                "REPEALS",
                "OUTGOING",
                "WHOLE_NORM",
                None,
                "REGEX",
                0.6,
                "snippet-a",
                "snippet-a",
                "exp-a",
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_2,
                "RG-LEGACY",
                10,
                "10",
                "texto",
                "",
                "relext-v2",
                "REPEALS",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.9,
                "snippet-b",
                "snippet-b",
                "exp-b",
                "2026-01-01T00:00:01Z",
            ),
        )
        conn.commit()

    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT target_norm_key, scope, method, confidence FROM relation_extractions"
        ).fetchall()

    assert rows == [("", "ARTICLE", "MIXED", 0.9)]
