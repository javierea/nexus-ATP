"""GraphRAG readiness validation KPIs and acceptance queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from rg_atp_pipeline.storage.migrations import ensure_schema


def run_graphrag_validation(
    db_path: Path,
    relation_extract_version: str | None = None,
    citation_extract_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate graph completeness and business-query usefulness."""
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        from_sql, where_sql, params = _relation_sql_parts(
            conn,
            relation_extract_version=relation_extract_version,
            citation_extract_version=citation_extract_version,
        )

        kpi_row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_external_relations,
                COUNT(DISTINCT re.source_doc_key) AS docs_with_external_relations,
                COUNT(DISTINCT re.relation_type) AS distinct_relation_types
            {from_sql}
            {where_sql}
            """,
            params,
        ).fetchone()

        intra_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_intra_norm_relations,
                COUNT(DISTINCT source_doc_key) AS docs_with_intra_norm_relations
            FROM intra_norm_relations
            """,
        ).fetchone()

        repeal_params = [*params, "REPEALS"]
        q_repeals = conn.execute(
            f"""
            SELECT COUNT(*)
            {from_sql}
            {_and_where(where_sql, 're.relation_type = ?')}
            """,
            repeal_params,
        ).fetchone()[0]
        q_modifies_and_according = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT re.source_doc_key
                {from_sql}
                {where_sql}
                GROUP BY re.source_doc_key
                HAVING SUM(CASE WHEN re.relation_type = 'MODIFIES' THEN 1 ELSE 0 END) > 0
                   AND SUM(CASE WHEN re.relation_type = 'ACCORDING_TO' THEN 1 ELSE 0 END) > 0
            ) q
            """,
            params,
        ).fetchone()[0]
        q_internal_coverage = conn.execute(
            "SELECT COUNT(DISTINCT source_doc_key) FROM intra_norm_relations"
        ).fetchone()[0]

    acceptance = {
        "has_external_relations": int(kpi_row[0] or 0) > 0,
        "has_intra_norm_relations": int(intra_row[0] or 0) > 0,
        "supports_repeal_queries": int(q_repeals or 0) > 0,
        "supports_cross_signal_queries": int(q_modifies_and_according or 0) > 0,
        "supports_internal_navigation": int(q_internal_coverage or 0) > 0,
    }

    return {
        "relation_extract_version_effective": relation_extract_version,
        "citation_extract_version_effective": citation_extract_version,
        "kpis": {
            "total_external_relations": int(kpi_row[0] or 0),
            "docs_with_external_relations": int(kpi_row[1] or 0),
            "distinct_relation_types": int(kpi_row[2] or 0),
            "total_intra_norm_relations": int(intra_row[0] or 0),
            "docs_with_intra_norm_relations": int(intra_row[1] or 0),
        },
        "business_queries": {
            "repeals_results": int(q_repeals or 0),
            "docs_with_modifies_and_according_to": int(q_modifies_and_according or 0),
            "docs_with_internal_navigation": int(q_internal_coverage or 0),
        },
        "acceptance": acceptance,
        "graphrag_ready": all(acceptance.values()),
    }


def _relation_sql_parts(
    conn: sqlite3.Connection,
    relation_extract_version: str | None,
    citation_extract_version: str | None,
) -> tuple[str, str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if relation_extract_version:
        where.append("re.extract_version = ?")
        params.append(relation_extract_version)
    if citation_extract_version:
        where.append("c.extract_version = ?")
        params.append(citation_extract_version)
    elif not relation_extract_version:
        row = conn.execute(
            "SELECT extract_version FROM relation_extractions ORDER BY created_at DESC, relation_id DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            where.append("re.extract_version = ?")
            params.append(str(row[0]))
    from_sql = "FROM relation_extractions re LEFT JOIN citations c ON c.citation_id = re.citation_id"
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    return from_sql, where_sql, params


def _and_where(where_sql: str, clause: str) -> str:
    if where_sql:
        return f"{where_sql} AND {clause}"
    return f"WHERE {clause}"
