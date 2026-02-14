"""SQLite migrations for norms catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def ensure_schema(db_path: Path) -> None:
    """Create norms catalog tables and indexes if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS norms (
                norm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                norm_key TEXT UNIQUE NOT NULL,
                norm_type TEXT NOT NULL,
                jurisdiction TEXT,
                year INTEGER,
                number TEXT,
                suffix TEXT,
                title TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS norm_aliases (
                alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                norm_id INTEGER NOT NULL,
                alias_text TEXT NOT NULL,
                alias_kind TEXT NOT NULL,
                confidence REAL NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(norm_id, alias_text),
                FOREIGN KEY(norm_id) REFERENCES norms(norm_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS norm_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                norm_id INTEGER NOT NULL,
                source_kind TEXT NOT NULL,
                source_method TEXT NOT NULL,
                url TEXT,
                is_authoritative INTEGER NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(norm_id) REFERENCES norms(norm_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS norm_source_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                UNIQUE(source_id, sha256),
                FOREIGN KEY(source_id) REFERENCES norm_sources(source_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_norms_norm_key ON norms(norm_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_norm_aliases_alias_text "
            "ON norm_aliases(alias_text)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_structure (
                doc_key TEXT PRIMARY KEY,
                structure_status TEXT NOT NULL,
                structure_confidence REAL,
                articles_detected INTEGER,
                annexes_detected INTEGER,
                notes TEXT,
                structured_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_key TEXT NOT NULL,
                unit_type TEXT NOT NULL,
                unit_number TEXT,
                title TEXT,
                text TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                start_line INTEGER,
                end_line INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
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
                detected_at TEXT NOT NULL,
                UNIQUE(
                    source_doc_key,
                    source_unit_id,
                    raw_text,
                    evidence_snippet,
                    extract_version
                )
            )
            """
        )
        citation_columns = {row[1] for row in conn.execute("PRAGMA table_info(citations)")}
        if "extract_version" not in citation_columns:
            conn.execute("ALTER TABLE citations RENAME TO citations_old")
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
                    detected_at TEXT NOT NULL,
                    evidence_unit_id INTEGER,
                    evidence_text TEXT,
                    evidence_kind TEXT,
                    match_start_in_unit INTEGER,
                    match_end_in_unit INTEGER,
                    UNIQUE(
                        source_doc_key,
                        source_unit_id,
                        raw_text,
                        evidence_snippet,
                        extract_version
                    )
                )
                """
            )
            old_columns = {row[1] for row in conn.execute("PRAGMA table_info(citations_old)")}
            selectable = [
                "citation_id",
                "source_doc_key",
                "source_unit_id",
                "source_unit_type",
                "raw_text",
                "norm_type_guess",
                "norm_key_candidate",
                "evidence_snippet",
                "regex_confidence",
                "detected_at",
            ]
            for optional in ["evidence_unit_id", "evidence_text", "evidence_kind"]:
                if optional in old_columns:
                    selectable.append(optional)
            select_sql = ", ".join(selectable)
            insert_sql = ", ".join(selectable + ["extract_version"])
            conn.execute(
                f"""
                INSERT INTO citations ({insert_sql})
                SELECT {select_sql}, 'citext-v1'
                FROM citations_old
                """
            )
            conn.execute("DROP TABLE citations_old")
            citation_columns = {row[1] for row in conn.execute("PRAGMA table_info(citations)")}
        if "evidence_unit_id" not in citation_columns:
            conn.execute("ALTER TABLE citations ADD COLUMN evidence_unit_id INTEGER")
        if "evidence_text" not in citation_columns:
            conn.execute("ALTER TABLE citations ADD COLUMN evidence_text TEXT")
        if "evidence_kind" not in citation_columns:
            conn.execute("ALTER TABLE citations ADD COLUMN evidence_kind TEXT")
        if "match_start_in_unit" not in citation_columns:
            conn.execute("ALTER TABLE citations ADD COLUMN match_start_in_unit INTEGER")
        if "match_end_in_unit" not in citation_columns:
            conn.execute("ALTER TABLE citations ADD COLUMN match_end_in_unit INTEGER")
        conn.execute(
            """
            UPDATE citations
            SET evidence_kind = COALESCE(evidence_kind, 'SNIPPET'),
                evidence_text = COALESCE(evidence_text, evidence_snippet)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_llm_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_id INTEGER NOT NULL,
                llm_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                is_reference INTEGER NOT NULL,
                norm_type TEXT NOT NULL,
                normalized_key TEXT,
                llm_confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(citation_id, llm_model, prompt_version),
                FOREIGN KEY(citation_id) REFERENCES citations(citation_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_links (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_id INTEGER NOT NULL,
                target_norm_id INTEGER,
                target_norm_key TEXT,
                resolution_status TEXT NOT NULL CHECK(
                    resolution_status IN (
                        'REJECTED',
                        'RESOLVED',
                        'PLACEHOLDER_CREATED',
                        'UNRESOLVED'
                    )
                ),
                resolution_confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(citation_id) REFERENCES citations(citation_id),
                FOREIGN KEY(target_norm_id) REFERENCES norms(norm_id)
            )
            """
        )
        conn.execute(
            """
            DELETE FROM citation_links
            WHERE link_id IN (
                SELECT old.link_id
                FROM citation_links AS old
                JOIN citation_links AS newest
                    ON newest.citation_id = old.citation_id
                   AND (
                        newest.created_at > old.created_at
                        OR (
                            newest.created_at = old.created_at
                            AND newest.link_id > old.link_id
                        )
                   )
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_citation_links_citation_id "
            "ON citation_links(citation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_doc_key "
            "ON citations(source_doc_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_norm_type "
            "ON citations(norm_type_guess)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation_links_target "
            "ON citation_links(target_norm_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_extractions (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_id INTEGER NOT NULL,
                link_id INTEGER,
                source_doc_key TEXT NOT NULL,
                target_norm_key TEXT,
                extract_version TEXT NOT NULL DEFAULT 'relext-v2',
                relation_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                method TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_snippet TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    citation_id,
                    link_id,
                    relation_type,
                    scope,
                    scope_detail,
                    method,
                    extract_version
                ),
                FOREIGN KEY(citation_id) REFERENCES citations(citation_id),
                FOREIGN KEY(link_id) REFERENCES citation_links(link_id)
            )
            """
        )
        relation_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(relation_extractions)")
        }
        relation_reviews_backup: list[tuple] = []
        if "extract_version" not in relation_columns:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='relation_llm_reviews'"
            ).fetchone():
                relation_reviews_backup = conn.execute(
                    """
                    SELECT
                        relation_id,
                        llm_model,
                        prompt_version,
                        relation_type,
                        direction,
                        scope,
                        scope_detail,
                        llm_confidence,
                        explanation,
                        created_at
                    FROM relation_llm_reviews
                    """
                ).fetchall()
                conn.execute("DROP TABLE relation_llm_reviews")
            conn.execute("PRAGMA defer_foreign_keys = ON")
            conn.execute("ALTER TABLE relation_extractions RENAME TO relation_extractions_old")
            conn.execute(
                """
                CREATE TABLE relation_extractions (
                    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    citation_id INTEGER NOT NULL,
                    link_id INTEGER,
                    source_doc_key TEXT NOT NULL,
                    target_norm_key TEXT,
                    extract_version TEXT NOT NULL DEFAULT 'relext-v2',
                    relation_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_detail TEXT,
                    method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_snippet TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_unit_id INTEGER,
                    source_unit_number TEXT,
                    source_unit_text TEXT,
                    extracted_match_snippet TEXT,
                    UNIQUE(
                        citation_id,
                        link_id,
                        relation_type,
                        scope,
                        scope_detail,
                        method,
                        extract_version
                    ),
                    FOREIGN KEY(citation_id) REFERENCES citations(citation_id),
                    FOREIGN KEY(link_id) REFERENCES citation_links(link_id)
                )
                """
            )
            old_rel_cols = {row[1] for row in conn.execute("PRAGMA table_info(relation_extractions_old)")}
            base = [
                "relation_id","citation_id","link_id","source_doc_key","target_norm_key",
                "relation_type","direction","scope","scope_detail","method","confidence",
                "evidence_snippet","explanation","created_at"
            ]
            for optional in ["source_unit_id", "source_unit_number", "source_unit_text", "extracted_match_snippet"]:
                if optional in old_rel_cols:
                    base.append(optional)
            select_base = [f"old.{column}" for column in base]
            conn.execute(
                f"""
                INSERT OR IGNORE INTO relation_extractions ({', '.join(base + ['extract_version'])})
                SELECT {', '.join(select_base)}, 'relext-v1'
                FROM relation_extractions_old AS old
                JOIN citations c ON c.citation_id = old.citation_id
                LEFT JOIN citation_links l ON l.link_id = old.link_id
                WHERE old.link_id IS NULL OR l.link_id IS NOT NULL
                """
            )
            conn.execute("DROP TABLE relation_extractions_old")
            relation_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(relation_extractions)")
            }
        if "source_unit_id" not in relation_columns:
            conn.execute("ALTER TABLE relation_extractions ADD COLUMN source_unit_id INTEGER")
        if "source_unit_number" not in relation_columns:
            conn.execute("ALTER TABLE relation_extractions ADD COLUMN source_unit_number TEXT")
        if "source_unit_text" not in relation_columns:
            conn.execute("ALTER TABLE relation_extractions ADD COLUMN source_unit_text TEXT")
        if "extracted_match_snippet" not in relation_columns:
            conn.execute(
                "ALTER TABLE relation_extractions ADD COLUMN extracted_match_snippet TEXT"
            )
        conn.execute(
            """
            UPDATE relation_extractions
            SET extracted_match_snippet = COALESCE(extracted_match_snippet, evidence_snippet)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_relation_extractions_unit_target_type
            ON relation_extractions(
                source_doc_key,
                source_unit_id,
                COALESCE(target_norm_key, ''),
                relation_type,
                scope,
                COALESCE(scope_detail, ''),
                method,
                extract_version
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_llm_reviews (
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
        if relation_reviews_backup:
            relation_ids = {
                row[0] for row in conn.execute("SELECT relation_id FROM relation_extractions")
            }
            relation_reviews_backup = [
                row for row in relation_reviews_backup if row[0] in relation_ids
            ]
            if relation_reviews_backup:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO relation_llm_reviews (
                        relation_id,
                        llm_model,
                        prompt_version,
                        relation_type,
                        direction,
                        scope,
                        scope_detail,
                        llm_confidence,
                        explanation,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    relation_reviews_backup,
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_extractions_source_doc_key "
            "ON relation_extractions(source_doc_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_extractions_target_norm_key "
            "ON relation_extractions(target_norm_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_extractions_relation_type "
            "ON relation_extractions(relation_type)"
        )
        conn.commit()
