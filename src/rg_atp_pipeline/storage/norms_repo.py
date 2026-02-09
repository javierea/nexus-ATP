"""Repository helpers for norms catalog."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rg_atp_pipeline.utils.normalize import normalize_text


@dataclass(frozen=True)
class NormRecord:
    """Norm record from the norms table."""

    norm_id: int
    norm_key: str
    norm_type: str
    jurisdiction: str | None
    year: int | None
    number: str | None
    suffix: str | None
    title: str | None
    status: str
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class NormsRepository:
    """SQLite-backed repository for norms catalog."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def upsert_norm(
        self,
        norm_key: str,
        norm_type: str,
        jurisdiction: str | None = None,
        year: int | None = None,
        number: str | None = None,
        suffix: str | None = None,
        title: str | None = None,
    ) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM norms WHERE norm_key = ?",
                (norm_key,),
            )
            row = cur.fetchone()
            if row:
                updated = {
                    "norm_type": norm_type or row["norm_type"],
                    "jurisdiction": jurisdiction or row["jurisdiction"],
                    "year": year if year is not None else row["year"],
                    "number": number or row["number"],
                    "suffix": suffix or row["suffix"],
                    "title": title or row["title"],
                    "updated_at": now,
                }
                conn.execute(
                    """
                    UPDATE norms
                    SET norm_type = ?,
                        jurisdiction = ?,
                        year = ?,
                        number = ?,
                        suffix = ?,
                        title = ?,
                        updated_at = ?
                    WHERE norm_key = ?
                    """,
                    (
                        updated["norm_type"],
                        updated["jurisdiction"],
                        updated["year"],
                        updated["number"],
                        updated["suffix"],
                        updated["title"],
                        updated["updated_at"],
                        norm_key,
                    ),
                )
                conn.commit()
                return int(row["norm_id"])
            conn.execute(
                """
                INSERT INTO norms (
                    norm_key,
                    norm_type,
                    jurisdiction,
                    year,
                    number,
                    suffix,
                    title,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    norm_key,
                    norm_type,
                    jurisdiction,
                    year,
                    number,
                    suffix,
                    title,
                    "KNOWN_ONLY",
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(
                conn.execute(
                    "SELECT norm_id FROM norms WHERE norm_key = ?",
                    (norm_key,),
                ).fetchone()[0]
            )

    def set_norm_status(self, norm_id: int, status: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE norms SET status = ?, updated_at = ? WHERE norm_id = ?",
                (status, now, norm_id),
            )
            conn.commit()

    def add_alias(
        self,
        norm_id: int,
        alias_text: str,
        alias_kind: str = "OTHER",
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO norm_aliases (
                    norm_id,
                    alias_text,
                    alias_kind,
                    confidence,
                    valid_from,
                    valid_to,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(norm_id, alias_text) DO UPDATE SET
                    alias_kind = excluded.alias_kind,
                    confidence = excluded.confidence,
                    valid_from = excluded.valid_from,
                    valid_to = excluded.valid_to
                """,
                (
                    norm_id,
                    alias_text,
                    alias_kind,
                    confidence,
                    valid_from,
                    valid_to,
                    now,
                ),
            )
            conn.commit()

    def resolve_norm_by_alias(
        self, text: str
    ) -> Optional[tuple[int, str, float, str]]:
        normalized = normalize_text(text)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT n.norm_id, n.norm_key, a.confidence, a.alias_text
                FROM norm_aliases a
                JOIN norms n ON n.norm_id = a.norm_id
                """
            ).fetchall()
        matches = [
            row
            for row in rows
            if normalize_text(row["alias_text"]) == normalized
        ]
        if not matches:
            return None
        matches.sort(
            key=lambda row: (row["confidence"], len(row["alias_text"])),
            reverse=True,
        )
        best = matches[0]
        return (
            int(best["norm_id"]),
            str(best["norm_key"]),
            float(best["confidence"]),
            str(best["alias_text"]),
        )

    def get_norm(self, norm_key: str) -> NormRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM norms WHERE norm_key = ?",
                (norm_key,),
            ).fetchone()
        if not row:
            return None
        return NormRecord(
            norm_id=int(row["norm_id"]),
            norm_key=str(row["norm_key"]),
            norm_type=str(row["norm_type"]),
            jurisdiction=row["jurisdiction"],
            year=row["year"],
            number=row["number"],
            suffix=row["suffix"],
            title=row["title"],
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_or_create_source(
        self,
        norm_id: int,
        source_kind: str,
        source_method: str,
        url: str | None,
        is_authoritative: bool,
        notes: str | None,
    ) -> int:
        now = _utc_now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT source_id
                FROM norm_sources
                WHERE norm_id = ?
                  AND source_kind = ?
                  AND source_method = ?
                  AND COALESCE(url, '') = COALESCE(?, '')
                """,
                (norm_id, source_kind, source_method, url),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE norm_sources
                    SET is_authoritative = ?,
                        notes = ?,
                        updated_at = ?
                    WHERE source_id = ?
                    """,
                    (int(is_authoritative), notes, now, int(row["source_id"])),
                )
                conn.commit()
                return int(row["source_id"])
            conn.execute(
                """
                INSERT INTO norm_sources (
                    norm_id,
                    source_kind,
                    source_method,
                    url,
                    is_authoritative,
                    notes,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    norm_id,
                    source_kind,
                    source_method,
                    url,
                    int(is_authoritative),
                    notes,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(
                conn.execute(
                    """
                    SELECT source_id
                    FROM norm_sources
                    WHERE norm_id = ?
                      AND source_kind = ?
                      AND source_method = ?
                      AND COALESCE(url, '') = COALESCE(?, '')
                    """,
                    (norm_id, source_kind, source_method, url),
                ).fetchone()[0]
            )

    def insert_source_version(
        self,
        source_id: int,
        sha256: str,
        downloaded_at: str,
        pdf_path: str,
        file_size_bytes: int,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT version_id
                FROM norm_source_versions
                WHERE source_id = ? AND sha256 = ?
                """,
                (source_id, sha256),
            ).fetchone()
            if row:
                return False
            conn.execute(
                """
                INSERT INTO norm_source_versions (
                    source_id,
                    sha256,
                    downloaded_at,
                    pdf_path,
                    file_size_bytes
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, sha256, downloaded_at, pdf_path, file_size_bytes),
            )
            conn.commit()
            return True
