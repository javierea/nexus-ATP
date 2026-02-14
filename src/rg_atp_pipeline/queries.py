"""Query helpers for rg_atp_pipeline UI."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def get_kpis(db_path: Path) -> dict[str, int]:
    """Return high-level pipeline counters."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(
                    DISTINCT CASE
                        WHEN d.doc_family = 'OLD' THEN printf('OLD-%s', d.number)
                        ELSE printf(
                            '%s-%s-%s',
                            d.doc_family,
                            COALESCE(d.year, ''),
                            d.number
                        )
                    END
                ) AS total_docs,
                SUM(CASE WHEN d.status = 'DOWNLOADED' THEN 1 ELSE 0 END) AS downloaded,
                COUNT(
                    DISTINCT CASE
                        WHEN d.status != 'MISSING' THEN NULL
                        WHEN d.doc_family = 'OLD' THEN printf('OLD-%s', d.number)
                        ELSE printf(
                            '%s-%s-%s',
                            d.doc_family,
                            COALESCE(d.year, ''),
                            d.number
                        )
                    END
                ) AS missing,
                SUM(CASE WHEN d.status = 'ERROR' THEN 1 ELSE 0 END) AS fetch_error,
                SUM(CASE WHEN COALESCE(d.text_status, 'NONE') = 'EXTRACTED'
                    THEN 1 ELSE 0 END) AS text_extracted,
                SUM(CASE WHEN COALESCE(d.text_status, 'NONE') = 'NEEDS_OCR'
                    THEN 1 ELSE 0 END) AS needs_ocr,
                SUM(CASE WHEN COALESCE(d.text_status, 'NONE') = 'ERROR'
                    THEN 1 ELSE 0 END) AS text_error,
                SUM(CASE WHEN s.structure_status = 'STRUCTURED'
                    THEN 1 ELSE 0 END) AS structured_ok,
                SUM(CASE WHEN s.structure_status = 'PARTIAL'
                    THEN 1 ELSE 0 END) AS partial,
                SUM(CASE WHEN s.structure_status = 'ERROR'
                    THEN 1 ELSE 0 END) AS structure_error
            FROM documents d
            LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
            """
        ).fetchone()
    if not row:
        return {}
    return {
        "total_docs": row[0] or 0,
        "downloaded": row[1] or 0,
        "missing": row[2] or 0,
        "error": row[3] or 0,
        "text_extracted": row[4] or 0,
        "needs_ocr": row[5] or 0,
        "text_error": row[6] or 0,
        "structured_ok": row[7] or 0,
        "partial": row[8] or 0,
        "structure_error": row[9] or 0,
    }


def get_confidence_distribution(db_path: Path) -> dict[str, int]:
    """Return confidence buckets for structured docs."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN structure_confidence >= 0.8 THEN 1 ELSE 0 END) AS high,
                SUM(CASE WHEN structure_confidence >= 0.6 AND structure_confidence < 0.8
                    THEN 1 ELSE 0 END) AS mid,
                SUM(CASE WHEN structure_confidence < 0.6 THEN 1 ELSE 0 END) AS low
            FROM doc_structure
            WHERE structure_confidence IS NOT NULL
            """
        ).fetchone()
    if not row:
        return {"high": 0, "mid": 0, "low": 0}
    return {"high": row[0] or 0, "mid": row[1] or 0, "low": row[2] or 0}


def list_documents(
    db_path: Path,
    filters: dict[str, Any],
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List documents with optional filters."""
    where, params = _build_filters(filters)
    sql = f"""
        SELECT
            d.doc_key,
            d.doc_family,
            d.year,
            d.number,
            d.status,
            d.http_status,
            d.last_checked_at,
            d.last_downloaded_at,
            d.url,
            d.error_message,
            COALESCE(d.text_status, 'NONE') AS text_status,
            d.text_extracted_at,
            d.char_count,
            d.pages_total,
            d.pages_with_text,
            d.alpha_ratio,
            s.structure_status,
            s.structure_confidence,
            s.articles_detected,
            s.structured_at
        FROM documents d
        LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
        WHERE {where}
        ORDER BY d.last_checked_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    with _connect(db_path) as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def list_backlog(db_path: Path, kind: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return backlog lists for the dashboard."""
    with _connect(db_path) as conn:
        if kind == "needs_ocr":
            cur = conn.execute(
                """
                SELECT
                    d.doc_key,
                    d.doc_family,
                    d.year,
                    d.number,
                    d.text_status,
                    d.char_count,
                    d.pages_total,
                    d.alpha_ratio,
                    d.text_extracted_at
                FROM documents d
                WHERE COALESCE(d.text_status, 'NONE') = 'NEEDS_OCR'
                ORDER BY d.text_extracted_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        elif kind == "partial_or_low_confidence":
            cur = conn.execute(
                """
                SELECT
                    d.doc_key,
                    d.doc_family,
                    d.year,
                    d.number,
                    s.structure_status,
                    s.structure_confidence,
                    s.articles_detected,
                    s.structured_at
                FROM documents d
                JOIN doc_structure s ON d.doc_key = s.doc_key
                WHERE s.structure_status = 'PARTIAL'
                   OR (s.structure_confidence IS NOT NULL AND s.structure_confidence < 0.6)
                ORDER BY s.structured_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        elif kind == "errors":
            cur = conn.execute(
                """
                SELECT
                    d.doc_key,
                    d.doc_family,
                    d.year,
                    d.number,
                    d.status,
                    COALESCE(d.text_status, 'NONE') AS text_status,
                    s.structure_status,
                    d.error_message,
                    d.last_checked_at
                FROM documents d
                LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
                WHERE d.status = 'ERROR'
                   OR COALESCE(d.text_status, 'NONE') = 'ERROR'
                   OR s.structure_status = 'ERROR'
                ORDER BY d.last_checked_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            return []
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def recent_activity(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent document activity based on latest timestamps."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT
                d.doc_key,
                d.doc_family,
                d.year,
                d.number,
                d.status,
                COALESCE(d.text_status, 'NONE') AS text_status,
                s.structure_status,
                MAX(
                    d.last_checked_at,
                    COALESCE(d.last_downloaded_at, ''),
                    COALESCE(d.text_extracted_at, ''),
                    COALESCE(s.structured_at, '')
                ) AS activity_at
            FROM documents d
            LEFT JOIN doc_structure s ON d.doc_key = s.doc_key
            GROUP BY d.doc_key
            ORDER BY activity_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def get_filter_options(db_path: Path) -> dict[str, list[Any]]:
    """Fetch distinct filter values for UI selects."""
    with _connect(db_path) as conn:
        statuses = [row[0] for row in conn.execute("SELECT DISTINCT status FROM documents")]
        families = [row[0] for row in conn.execute("SELECT DISTINCT doc_family FROM documents")]
        years = [row[0] for row in conn.execute("SELECT DISTINCT year FROM documents")]
        text_statuses = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT COALESCE(text_status, 'NONE') FROM documents"
            )
        ]
        structure_statuses = [
            row[0] for row in conn.execute("SELECT DISTINCT structure_status FROM doc_structure")
        ]
    return {
        "statuses": sorted({s for s in statuses if s}),
        "doc_families": sorted({s for s in families if s}),
        "years": sorted({y for y in years if y is not None}),
        "text_statuses": sorted({s for s in text_statuses if s}),
        "structure_statuses": sorted({s for s in structure_statuses if s}),
    }



def get_structure_summary(db_path: Path) -> dict[str, Any]:
    """Return aggregate metrics for structure stage dashboards."""
    if not db_path.exists():
        return {
            "total_docs": 0,
            "structured_ok": 0,
            "partial": 0,
            "error": 0,
            "without_structure": 0,
            "units_total": 0,
            "articles_total": 0,
            "avg_confidence": None,
            "last_structured_at": None,
        }

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(d.doc_key) AS total_docs,
                SUM(CASE WHEN ds.structure_status = 'STRUCTURED' THEN 1 ELSE 0 END) AS structured_ok,
                SUM(CASE WHEN ds.structure_status = 'PARTIAL' THEN 1 ELSE 0 END) AS partial,
                SUM(CASE WHEN ds.structure_status = 'ERROR' THEN 1 ELSE 0 END) AS error,
                SUM(CASE WHEN ds.doc_key IS NULL THEN 1 ELSE 0 END) AS without_structure,
                COALESCE(SUM(ds.articles_detected), 0) AS articles_total,
                AVG(ds.structure_confidence) AS avg_confidence,
                MAX(ds.structured_at) AS last_structured_at
            FROM documents d
            LEFT JOIN doc_structure ds ON ds.doc_key = d.doc_key
            """
        ).fetchone()
        units_total = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]

    if not row:
        return {
            "total_docs": 0,
            "structured_ok": 0,
            "partial": 0,
            "error": 0,
            "without_structure": 0,
            "units_total": 0,
            "articles_total": 0,
            "avg_confidence": None,
            "last_structured_at": None,
        }

    return {
        "total_docs": row[0] or 0,
        "structured_ok": row[1] or 0,
        "partial": row[2] or 0,
        "error": row[3] or 0,
        "without_structure": row[4] or 0,
        "units_total": units_total or 0,
        "articles_total": row[5] or 0,
        "avg_confidence": row[6],
        "last_structured_at": row[7],
    }


def get_structure_anomalies(
    db_path: Path,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return suspicious structure records to audit from UI filters."""
    if not db_path.exists():
        return []

    payload = filters or {}
    where_clauses = [
        "("
        "ds.structure_status IN ('PARTIAL', 'ERROR') "
        "OR ds.structure_confidence < ? "
        "OR (ds.structure_status = 'STRUCTURED' AND COALESCE(ds.articles_detected, 0) = 0)"
        ")"
    ]
    params: list[Any] = [float(payload.get("confidence_lt", 0.6))]

    if payload.get("doc_family"):
        where_clauses.append("d.doc_family = ?")
        params.append(payload["doc_family"])
    if payload.get("year") is not None:
        where_clauses.append("d.year = ?")
        params.append(int(payload["year"]))
    if payload.get("structure_status"):
        where_clauses.append("ds.structure_status = ?")
        params.append(payload["structure_status"])

    limit = int(payload.get("limit", 200))
    params.append(max(1, min(limit, 2000)))

    query = f"""
        SELECT
            d.doc_key,
            d.doc_family,
            d.year,
            COALESCE(d.text_status, 'NONE') AS text_status,
            ds.structure_status,
            ds.structure_confidence,
            ds.articles_detected,
            ds.annexes_detected,
            ds.notes,
            ds.structured_at
        FROM documents d
        JOIN doc_structure ds ON ds.doc_key = d.doc_key
        WHERE {' AND '.join(where_clauses)}
        ORDER BY ds.structured_at DESC, d.doc_key ASC
        LIMIT ?
    """

    with _connect(db_path) as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def get_units_for_doc(db_path: Path, doc_key: str) -> list[dict[str, Any]]:
    """Return units generated for one document key."""
    if not db_path.exists() or not doc_key.strip():
        return []

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT
                id,
                doc_key,
                unit_type,
                unit_number,
                title,
                start_line,
                end_line,
                start_char,
                end_char,
                LENGTH(text) AS text_length,
                substr(text, 1, 320) AS text_preview,
                created_at
            FROM units
            WHERE doc_key = ?
            ORDER BY id ASC
            """,
            (doc_key.strip(),),
        )
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]

def _build_filters(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if status := filters.get("status"):
        where.append("d.status = ?")
        params.append(status)
    if doc_family := filters.get("doc_family"):
        where.append("d.doc_family = ?")
        params.append(doc_family)
    if year := filters.get("year"):
        where.append("d.year = ?")
        params.append(year)
    if text_status := filters.get("text_status"):
        where.append("COALESCE(d.text_status, 'NONE') = ?")
        params.append(text_status)
    if structure_status := filters.get("structure_status"):
        where.append("s.structure_status = ?")
        params.append(structure_status)
    if filters.get("needs_ocr"):
        where.append("COALESCE(d.text_status, 'NONE') = 'NEEDS_OCR'")
    if confidence_lt := filters.get("confidence_lt"):
        where.append("s.structure_confidence < ?")
        params.append(confidence_lt)
    if confidence_gte := filters.get("confidence_gte"):
        where.append("s.structure_confidence >= ?")
        params.append(confidence_gte)
    if search := filters.get("search"):
        where.append("d.doc_key LIKE ?")
        params.append(f"%{search}%")
    clause = " AND ".join(where) if where else "1=1"
    return clause, params


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)
