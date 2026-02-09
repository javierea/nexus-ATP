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
        conn.commit()
