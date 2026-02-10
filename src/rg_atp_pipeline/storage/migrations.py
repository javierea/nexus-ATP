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
            CREATE TABLE IF NOT EXISTS citations (
                citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_key TEXT NOT NULL,
                source_unit_id TEXT,
                source_unit_type TEXT,
                raw_text TEXT NOT NULL,
                norm_type_guess TEXT NOT NULL,
                norm_key_candidate TEXT,
                evidence_snippet TEXT NOT NULL,
                regex_confidence REAL NOT NULL,
                detected_at TEXT NOT NULL,
                UNIQUE(
                    source_doc_key,
                    source_unit_id,
                    raw_text,
                    evidence_snippet
                )
            )
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
        conn.commit()
