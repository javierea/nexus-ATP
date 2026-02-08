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
    text_status: str
    text_path: str | None
    text_extracted_at: str | None
    char_count: int | None
    pages_total: int | None
    pages_with_text: int | None
    alpha_ratio: float | None
    structure_status: str | None = None
    structure_confidence: float | None = None
    articles_detected: int | None = None
    annexes_detected: int | None = None


@dataclass(frozen=True)
class DocumentLookup:
    """Minimal document lookup for skip checks."""

    status: str
    latest_pdf_path: str | None
    latest_sha256: str | None


@dataclass(frozen=True)
class DocStructureRecord:
    """Structured document metadata."""

    doc_key: str
    structure_status: str
    structure_confidence: float | None
    articles_detected: int | None
    annexes_detected: int | None
    notes: str | None
    structured_at: str | None


@dataclass(frozen=True)
class UnitRecord:
    """Structured unit record."""

    id: int
    doc_key: str
    unit_type: str
    unit_number: str | None
    title: str | None
    text: str
    start_char: int | None
    end_char: int | None
    start_line: int | None
    end_line: int | None
    created_at: str


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
                    error_message TEXT,
                    text_status TEXT,
                    text_path TEXT,
                    text_extracted_at TEXT,
                    char_count INTEGER,
                    pages_total INTEGER,
                    pages_with_text INTEGER,
                    alpha_ratio REAL
                )
                """
            )
            conn.commit()
        self.migrate()

    def migrate(self) -> None:
        """Ensure schema includes text extraction columns."""
        with self._connect() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
            to_add = [
                ("text_status", "TEXT"),
                ("text_path", "TEXT"),
                ("text_extracted_at", "TEXT"),
                ("char_count", "INTEGER"),
                ("pages_total", "INTEGER"),
                ("pages_with_text", "INTEGER"),
                ("alpha_ratio", "REAL"),
            ]
            for column, col_type in to_add:
                if column not in existing:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {column} {col_type}")
            conn.execute(
                "UPDATE documents SET text_status = 'NONE' WHERE text_status IS NULL"
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
                    d.doc_key,
                    d.url,
                    d.doc_family,
                    d.year,
                    d.number,
                    d.first_seen_at,
                    d.last_checked_at,
                    d.last_downloaded_at,
                    d.latest_sha256,
                    d.latest_pdf_path,
                    d.status,
                    d.http_status,
                    d.error_message,
                    d.text_status,
                    d.text_path,
                    d.text_extracted_at,
                    d.char_count,
                    d.pages_total,
                    d.pages_with_text,
                    d.alpha_ratio,
                    s.structure_status,
                    s.structure_confidence,
                    s.articles_detected,
                    s.annexes_detected
                FROM documents d
                LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
                ORDER BY d.last_checked_at DESC
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
                text_status=row[13] or "NONE",
                text_path=row[14],
                text_extracted_at=row[15],
                char_count=row[16],
                pages_total=row[17],
                pages_with_text=row[18],
                alpha_ratio=row[19],
                structure_status=row[20],
                structure_confidence=row[21],
                articles_detected=row[22],
                annexes_detected=row[23],
            )
            for row in rows
        ]

    def get_document(self, doc_key: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    d.doc_key,
                    d.url,
                    d.doc_family,
                    d.year,
                    d.number,
                    d.first_seen_at,
                    d.last_checked_at,
                    d.last_downloaded_at,
                    d.latest_sha256,
                    d.latest_pdf_path,
                    d.status,
                    d.http_status,
                    d.error_message,
                    d.text_status,
                    d.text_path,
                    d.text_extracted_at,
                    d.char_count,
                    d.pages_total,
                    d.pages_with_text,
                    d.alpha_ratio,
                    s.structure_status,
                    s.structure_confidence,
                    s.articles_detected,
                    s.annexes_detected
                FROM documents d
                LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
                WHERE d.doc_key = ?
                """,
                (doc_key,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return DocumentRecord(
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
            text_status=row[13] or "NONE",
            text_path=row[14],
            text_extracted_at=row[15],
            char_count=row[16],
            pages_total=row[17],
            pages_with_text=row[18],
            alpha_ratio=row[19],
            structure_status=row[20],
            structure_confidence=row[21],
            articles_detected=row[22],
            annexes_detected=row[23],
        )

    def list_text_candidates(
        self,
        status: str | None,
        limit: int | None,
        doc_key: str | None,
        only_text_status: str | None,
    ) -> list[DocumentRecord]:
        where = []
        params: list[object] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if doc_key:
            where.append("doc_key = ?")
            params.append(doc_key)
        if only_text_status:
            where.append("COALESCE(text_status, 'NONE') = ?")
            params.append(only_text_status)
        clause = " AND ".join(where) if where else "1=1"
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
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
                    error_message,
                    text_status,
                    text_path,
                    text_extracted_at,
                    char_count,
                    pages_total,
                    pages_with_text,
                    alpha_ratio
                FROM documents
                WHERE {clause}
                ORDER BY last_checked_at DESC
                {limit_clause}
                """,
                params,
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
                text_status=row[13] or "NONE",
                text_path=row[14],
                text_extracted_at=row[15],
                char_count=row[16],
                pages_total=row[17],
                pages_with_text=row[18],
                alpha_ratio=row[19],
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
                    error_message,
                    text_status,
                    text_path,
                    text_extracted_at,
                    char_count,
                    pages_total,
                    pages_with_text,
                    alpha_ratio
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    error_message = excluded.error_message,
                    text_status = COALESCE(documents.text_status, excluded.text_status),
                    text_path = COALESCE(documents.text_path, excluded.text_path),
                    text_extracted_at = COALESCE(
                        documents.text_extracted_at,
                        excluded.text_extracted_at
                    ),
                    char_count = COALESCE(documents.char_count, excluded.char_count),
                    pages_total = COALESCE(documents.pages_total, excluded.pages_total),
                    pages_with_text = COALESCE(
                        documents.pages_with_text,
                        excluded.pages_with_text
                    ),
                    alpha_ratio = COALESCE(documents.alpha_ratio, excluded.alpha_ratio)
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
                    record.text_status,
                    record.text_path,
                    record.text_extracted_at,
                    record.char_count,
                    record.pages_total,
                    record.pages_with_text,
                    record.alpha_ratio,
                ),
            )
            conn.commit()

    def update_text_info(
        self,
        doc_key: str,
        text_status: str,
        text_path: str | None,
        text_extracted_at: str | None,
        char_count: int | None,
        pages_total: int | None,
        pages_with_text: int | None,
        alpha_ratio: float | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET
                    text_status = ?,
                    text_path = ?,
                    text_extracted_at = ?,
                    char_count = ?,
                    pages_total = ?,
                    pages_with_text = ?,
                    alpha_ratio = ?
                WHERE doc_key = ?
                """,
                (
                    text_status,
                    text_path,
                    text_extracted_at,
                    char_count,
                    pages_total,
                    pages_with_text,
                    alpha_ratio,
                    doc_key,
                ),
            )
            conn.commit()

    def upsert_doc_structure(
        self,
        record: DocStructureRecord,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO doc_structure (
                    doc_key,
                    structure_status,
                    structure_confidence,
                    articles_detected,
                    annexes_detected,
                    notes,
                    structured_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_key) DO UPDATE SET
                    structure_status = excluded.structure_status,
                    structure_confidence = excluded.structure_confidence,
                    articles_detected = excluded.articles_detected,
                    annexes_detected = excluded.annexes_detected,
                    notes = excluded.notes,
                    structured_at = excluded.structured_at
                """,
                (
                    record.doc_key,
                    record.structure_status,
                    record.structure_confidence,
                    record.articles_detected,
                    record.annexes_detected,
                    record.notes,
                    record.structured_at,
                ),
            )
            conn.commit()

    def get_doc_structure(self, doc_key: str) -> DocStructureRecord | None:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    doc_key,
                    structure_status,
                    structure_confidence,
                    articles_detected,
                    annexes_detected,
                    notes,
                    structured_at
                FROM doc_structure
                WHERE doc_key = ?
                """,
                (doc_key,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return DocStructureRecord(
            doc_key=row[0],
            structure_status=row[1],
            structure_confidence=row[2],
            articles_detected=row[3],
            annexes_detected=row[4],
            notes=row[5],
            structured_at=row[6],
        )

    def delete_units(self, doc_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM units WHERE doc_key = ?", (doc_key,))
            conn.commit()

    def insert_units(self, units: list[UnitRecord]) -> None:
        if not units:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO units (
                    doc_key,
                    unit_type,
                    unit_number,
                    title,
                    text,
                    start_char,
                    end_char,
                    start_line,
                    end_line,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        unit.doc_key,
                        unit.unit_type,
                        unit.unit_number,
                        unit.title,
                        unit.text,
                        unit.start_char,
                        unit.end_char,
                        unit.start_line,
                        unit.end_line,
                        unit.created_at,
                    )
                    for unit in units
                ],
            )
            conn.commit()

    def list_units(self, doc_key: str) -> list[UnitRecord]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    id,
                    doc_key,
                    unit_type,
                    unit_number,
                    title,
                    text,
                    start_char,
                    end_char,
                    start_line,
                    end_line,
                    created_at
                FROM units
                WHERE doc_key = ?
                ORDER BY id
                """,
                (doc_key,),
            )
            rows = cur.fetchall()
        return [
            UnitRecord(
                id=row[0],
                doc_key=row[1],
                unit_type=row[2],
                unit_number=row[3],
                title=row[4],
                text=row[5],
                start_char=row[6],
                end_char=row[7],
                start_line=row[8],
                end_line=row[9],
                created_at=row[10],
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
