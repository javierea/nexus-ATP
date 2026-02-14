"""Transactional merge of norm references from one key to another."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rg_atp_pipeline.storage.migrations import ensure_schema


MERGE_TRACE_ALIAS_KIND = "MERGE_TRACE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _resolve_norm_id(conn: sqlite3.Connection, norm_key: str) -> int | None:
    row = conn.execute(
        "SELECT norm_id FROM norms WHERE norm_key = ?",
        (norm_key,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


def merge_norm(
    db_path: Path,
    from_norm_key: str,
    to_norm_key: str,
    apply: bool,
) -> dict[str, Any]:
    """Merge references from ``from_norm_key`` into ``to_norm_key``."""
    ensure_schema(db_path)

    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "from_norm_key": from_norm_key,
        "to_norm_key": to_norm_key,
        "from_norm_id": None,
        "to_norm_id": None,
        "rows_affected": {
            "citation_links": 0,
            "relation_extractions": 0,
            "norm_aliases": 0,
            "norm_sources": 0,
            "norm_source_versions": 0,
        },
        "aliases_moved": {
            "inserted_to_target": 0,
            "deleted_from_source": 0,
            "trace_alias_added": 0,
        },
        "errors": [],
    }

    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")

    try:
        from_norm_id = _resolve_norm_id(conn, from_norm_key)
        to_norm_id = _resolve_norm_id(conn, to_norm_key)
        summary["from_norm_id"] = from_norm_id
        summary["to_norm_id"] = to_norm_id

        if from_norm_id is None:
            summary["errors"].append(f"Norma origen no encontrada: {from_norm_key}")
        if to_norm_id is None:
            summary["errors"].append(f"Norma destino no encontrada: {to_norm_key}")
        if from_norm_id is not None and to_norm_id is not None and from_norm_id == to_norm_id:
            summary["errors"].append("Norma origen y destino son la misma.")

        if summary["errors"]:
            return summary

        summary["rows_affected"]["citation_links"] = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM citation_links
                WHERE target_norm_key = ?
                   OR target_norm_id = ?
                """,
                (from_norm_key, from_norm_id),
            ).fetchone()[0]
        )
        summary["rows_affected"]["relation_extractions"] = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM relation_extractions
                WHERE target_norm_key = ?
                """,
                (from_norm_key,),
            ).fetchone()[0]
        )
        summary["rows_affected"]["norm_aliases"] = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM norm_aliases
                WHERE norm_id = ?
                """,
                (from_norm_id,),
            ).fetchone()[0]
        )

        if _table_exists(conn, "norm_sources"):
            summary["rows_affected"]["norm_sources"] = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM norm_sources
                    WHERE norm_id = ?
                    """,
                    (from_norm_id,),
                ).fetchone()[0]
            )

        if _table_exists(conn, "norm_source_versions") and _table_exists(conn, "norm_sources"):
            summary["rows_affected"]["norm_source_versions"] = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM norm_source_versions v
                    JOIN norm_sources s ON s.source_id = v.source_id
                    WHERE s.norm_id = ?
                    """,
                    (from_norm_id,),
                ).fetchone()[0]
            )

        if not apply:
            return summary

        conn.execute("BEGIN IMMEDIATE")

        update_links = conn.execute(
            """
            UPDATE citation_links
            SET target_norm_id = ?,
                target_norm_key = ?
            WHERE target_norm_key = ?
               OR target_norm_id = ?
            """,
            (to_norm_id, to_norm_key, from_norm_key, from_norm_id),
        )
        summary["rows_affected"]["citation_links"] = update_links.rowcount

        update_relations = conn.execute(
            """
            UPDATE relation_extractions
            SET target_norm_key = ?
            WHERE target_norm_key = ?
            """,
            (to_norm_key, from_norm_key),
        )
        summary["rows_affected"]["relation_extractions"] = update_relations.rowcount

        aliases = conn.execute(
            """
            SELECT alias_text, alias_kind, confidence, valid_from, valid_to, created_at
            FROM norm_aliases
            WHERE norm_id = ?
            """,
            (from_norm_id,),
        ).fetchall()

        inserted_aliases = 0
        for alias in aliases:
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO norm_aliases (
                    norm_id,
                    alias_text,
                    alias_kind,
                    confidence,
                    valid_from,
                    valid_to,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    to_norm_id,
                    alias["alias_text"],
                    alias["alias_kind"],
                    alias["confidence"],
                    alias["valid_from"],
                    alias["valid_to"],
                    alias["created_at"],
                ),
            ).rowcount
            inserted_aliases += inserted

        summary["aliases_moved"]["inserted_to_target"] = inserted_aliases

        deleted_aliases = conn.execute(
            "DELETE FROM norm_aliases WHERE norm_id = ?",
            (from_norm_id,),
        ).rowcount
        summary["aliases_moved"]["deleted_from_source"] = deleted_aliases
        summary["rows_affected"]["norm_aliases"] = deleted_aliases

        trace_added = conn.execute(
            """
            INSERT OR IGNORE INTO norm_aliases (
                norm_id,
                alias_text,
                alias_kind,
                confidence,
                valid_from,
                valid_to,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (to_norm_id, from_norm_key, MERGE_TRACE_ALIAS_KIND, 1.0, None, None, _utc_now()),
        ).rowcount
        summary["aliases_moved"]["trace_alias_added"] = trace_added

        if _table_exists(conn, "norm_sources"):
            moved_sources = conn.execute(
                "UPDATE norm_sources SET norm_id = ? WHERE norm_id = ?",
                (to_norm_id, from_norm_id),
            ).rowcount
            summary["rows_affected"]["norm_sources"] = moved_sources

        conn.commit()
        return summary
    except sqlite3.DatabaseError as exc:
        conn.rollback()
        summary["errors"].append(str(exc))
        return summary
    finally:
        conn.close()
