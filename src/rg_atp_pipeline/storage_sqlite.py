"""SQLite storage for rg_atp_pipeline documents."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DocumentRecord:
    """Document metadata for persistence."""

    doc_key: str
    url: str
    doc_family: str
    year: int | None
    number: int | None
    first_seen_at: str
    last_checked_at: str
    last_downloaded_at: str | None
    latest_sha256: str | None
    latest_pdf_path: str | None
    status: str
    http_status: int | None
    error_message: str | None


@dataclass(frozen=True)
class DocumentLookup:
    """Minimal document lookup for skip checks."""

    status: str
    latest_pdf_path: str | None
    latest_sha256: str | None


class DocumentStore:
    """SQLite-backed document registry."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_key TEXT UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    doc_family TEXT NOT NULL,
                    year INTEGER,
                    number INTEGER,
                    first_seen_at TEXT NOT NULL,
                    last_checked_at TEXT NOT NULL,
                    last_downloaded_at TEXT,
                    latest_sha256 TEXT,
                    latest_pdf_path TEXT,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    error_message TEXT
                )
                """
            )
            conn.commit()

    def get_latest_sha(self, doc_key: str) -> Optional[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT latest_sha256 FROM documents WHERE doc_key = ?",
                (doc_key,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def get_record(self, doc_key: str) -> Optional[DocumentLookup]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT status, latest_pdf_path, latest_sha256
                FROM documents
                WHERE doc_key = ?
                """,
                (doc_key,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return DocumentLookup(status=row[0], latest_pdf_path=row[1], latest_sha256=row[2])

    def list_records(self, limit: int = 200) -> list[DocumentRecord]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    doc_key,
                    url,
                    doc_family,
                    year,
                    number,
                    first_seen_at,
                    last_checked_at,
                    last_downloaded_at,
                    latest_sha256,
                    latest_pdf_path,
                    status,
                    http_status,
                    error_message
                FROM documents
                ORDER BY last_checked_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            DocumentRecord(
                doc_key=row[0],
                url=row[1],
                doc_family=row[2],
                year=row[3],
                number=row[4],
                first_seen_at=row[5],
                last_checked_at=row[6],
                last_downloaded_at=row[7],
                latest_sha256=row[8],
                latest_pdf_path=row[9],
                status=row[10],
                http_status=row[11],
                error_message=row[12],
            )
            for row in rows
        ]

    def delete_record(self, doc_key: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_key = ?", (doc_key,))
            conn.commit()
            return cur.rowcount

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_key,
                    url,
                    doc_family,
                    year,
                    number,
                    first_seen_at,
                    last_checked_at,
                    last_downloaded_at,
                    latest_sha256,
                    latest_pdf_path,
                    status,
                    http_status,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_key) DO UPDATE SET
                    url = excluded.url,
                    doc_family = excluded.doc_family,
                    year = excluded.year,
                    number = excluded.number,
                    first_seen_at = COALESCE(documents.first_seen_at, excluded.first_seen_at),
                    last_checked_at = excluded.last_checked_at,
                    last_downloaded_at = excluded.last_downloaded_at,
                    latest_sha256 = excluded.latest_sha256,
                    latest_pdf_path = excluded.latest_pdf_path,
                    status = excluded.status,
                    http_status = excluded.http_status,
                    error_message = excluded.error_message
                """,
                (
                    record.doc_key,
                    record.url,
                    record.doc_family,
                    record.year,
                    record.number,
                    record.first_seen_at,
                    record.last_checked_at,
                    record.last_downloaded_at,
                    record.latest_sha256,
                    record.latest_pdf_path,
                    record.status,
                    record.http_status,
                    record.error_message,
                ),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
