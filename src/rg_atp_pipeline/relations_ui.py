"""UI-facing helpers for Etapa 4.1 (relations)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from rg_atp_pipeline.services.relations_service import run_relations


def run_relations_ui(
    db_path: Path,
    data_dir: Path,
    doc_keys: list[str] | None,
    limit_docs: int | None,
    llm_mode: str,
    min_confidence: float,
    prompt_version: str,
    batch_size: int | None,
    ollama_model: str | None,
    ollama_base_url: str | None,
) -> dict[str, Any]:
    """Execute Stage 4.1 relation extraction and return summary payload."""
    effective_batch_size = int(batch_size) if batch_size is not None else 20
    return run_relations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=doc_keys,
        limit_docs=limit_docs,
        llm_mode=llm_mode,
        min_confidence=min_confidence,
        prompt_version=prompt_version,
        batch_size=effective_batch_size,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )


def get_relations_summary(db_path: Path, prompt_version: str | None = None) -> dict[str, Any]:
    """Return aggregate metrics for relation extractions and LLM reviews."""
    if not db_path.exists():
        return {
            "total_relations": 0,
            "by_type": {},
            "by_scope": {},
            "by_direction": {},
            "last_created_at": None,
            "last_prompt_version": None,
            "last_llm_model": None,
            "llm_reviews_count": 0,
            "docs_covered": 0,
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        base_where, params = _relations_prompt_filter(prompt_version)

        total_row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_relations,
                COUNT(DISTINCT source_doc_key) AS docs_covered,
                MAX(created_at) AS last_created_at
            FROM relation_extractions
            {base_where}
            """,
            params,
        ).fetchone()

        by_type = _group_count(conn, "relation_type", base_where, params)
        by_scope = _group_count(conn, "scope", base_where, params)
        by_direction = _group_count(conn, "direction", base_where, params)

        review_where = ""
        review_params: list[Any] = []
        if prompt_version:
            review_where = "WHERE prompt_version = ?"
            review_params.append(prompt_version)

        review_row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS llm_reviews_count,
                MAX(prompt_version) AS last_prompt_version,
                MAX(llm_model) AS last_llm_model
            FROM relation_llm_reviews
            {review_where}
            """,
            review_params,
        ).fetchone()

    return {
        "total_relations": int(total_row["total_relations"] or 0),
        "by_type": by_type,
        "by_scope": by_scope,
        "by_direction": by_direction,
        "last_created_at": total_row["last_created_at"],
        "last_prompt_version": review_row["last_prompt_version"],
        "last_llm_model": review_row["last_llm_model"],
        "llm_reviews_count": int(review_row["llm_reviews_count"] or 0),
        "docs_covered": int(total_row["docs_covered"] or 0),
    }


def get_relations_table(
    db_path: Path,
    prompt_version: str | None = None,
    relation_type: str | None = None,
    scope: str | None = None,
    limit: int = 500,
):
    """Return relation rows for exploration in a dataframe."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    empty_columns = [
        "relation_id",
        "source_doc_key",
        "target_norm_key",
        "relation_type",
        "direction",
        "scope",
        "scope_detail",
        "confidence",
        "method",
        "created_at",
        "evidence_snippet",
        "explanation",
        "llm_explanation",
        "llm_confidence",
        "llm_model",
        "prompt_version",
    ]

    if not db_path.exists():
        return pd.DataFrame(columns=empty_columns) if pd else []

    where_clauses: list[str] = []
    params: list[Any] = []
    if prompt_version:
        join = """
            LEFT JOIN (
                SELECT
                    relation_id,
                    explanation,
                    llm_confidence,
                    llm_model,
                    prompt_version
                FROM (
                    SELECT
                        relation_id,
                        explanation,
                        llm_confidence,
                        llm_model,
                        prompt_version,
                        ROW_NUMBER() OVER (
                            PARTITION BY relation_id
                            ORDER BY created_at DESC, review_id DESC
                        ) AS rn
                    FROM relation_llm_reviews
                    WHERE prompt_version = ?
                )
                WHERE rn = 1
            ) rv ON rv.relation_id = re.relation_id
        """
        params.append(prompt_version)
    else:
        join = """
            LEFT JOIN (
                SELECT
                    relation_id,
                    explanation,
                    llm_confidence,
                    llm_model,
                    prompt_version
                FROM (
                    SELECT
                        relation_id,
                        explanation,
                        llm_confidence,
                        llm_model,
                        prompt_version,
                        ROW_NUMBER() OVER (
                            PARTITION BY relation_id
                            ORDER BY created_at DESC, review_id DESC
                        ) AS rn
                    FROM relation_llm_reviews
                )
                WHERE rn = 1
            ) rv ON rv.relation_id = re.relation_id
        """
    if relation_type:
        where_clauses.append("re.relation_type = ?")
        params.append(relation_type)
    if scope:
        where_clauses.append("re.scope = ?")
        params.append(scope)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(max(1, min(limit, 5000)))

    query = f"""
        SELECT
            re.relation_id,
            re.source_doc_key,
            re.target_norm_key,
            re.relation_type,
            re.direction,
            re.scope,
            re.scope_detail,
            re.confidence,
            re.method,
            re.created_at,
            re.evidence_snippet,
            re.explanation,
            rv.explanation AS llm_explanation,
            rv.llm_confidence,
            rv.llm_model,
            rv.prompt_version
        FROM relation_extractions re
        {join}
        {where_sql}
        ORDER BY re.created_at DESC, re.relation_id DESC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    records = [dict(row) for row in rows]
    return pd.DataFrame(records) if pd else records


def get_relations_qa_samples(db_path: Path, relation_type: str, n: int = 30):
    """Return random sample of relations by type for QA auditing."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if not db_path.exists() or not relation_type:
        return pd.DataFrame() if pd else []

    query = """
        SELECT
            relation_id,
            source_doc_key,
            target_norm_key,
            relation_type,
            direction,
            scope,
            scope_detail,
            confidence,
            method,
            created_at,
            evidence_snippet,
            explanation
        FROM relation_extractions
        WHERE relation_type = ?
        ORDER BY RANDOM()
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (relation_type, max(1, min(n, 500)))).fetchall()
    records = [dict(row) for row in rows]
    return pd.DataFrame(records) if pd else records


def get_relations_inconsistencies(db_path: Path) -> dict[str, Any]:
    """Compute simple consistency checks over extracted relations."""
    if not db_path.exists():
        return {
            "count_effect_without_target": 0,
            "count_article_without_detail": 0,
            "count_unknown_high_conf": 0,
            "top_explanations": [],
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN COALESCE(TRIM(target_norm_key), '') = '' THEN 1 ELSE 0 END)
                    AS count_effect_without_target,
                SUM(
                    CASE
                        WHEN scope = 'ARTICLE'
                         AND COALESCE(TRIM(scope_detail), '') = ''
                        THEN 1 ELSE 0
                    END
                ) AS count_article_without_detail,
                SUM(
                    CASE
                        WHEN relation_type = 'UNKNOWN'
                         AND confidence >= 0.8
                        THEN 1 ELSE 0
                    END
                ) AS count_unknown_high_conf
            FROM relation_extractions
            """
        ).fetchone()
        explanation_rows = conn.execute(
            """
            SELECT explanation, COUNT(*) AS total
            FROM relation_extractions
            GROUP BY explanation
            ORDER BY total DESC, explanation ASC
            LIMIT 10
            """
        ).fetchall()

    return {
        "count_effect_without_target": int(row["count_effect_without_target"] or 0),
        "count_article_without_detail": int(row["count_article_without_detail"] or 0),
        "count_unknown_high_conf": int(row["count_unknown_high_conf"] or 0),
        "top_explanations": [
            {"explanation": exp["explanation"], "count": int(exp["total"])}
            for exp in explanation_rows
        ],
    }


def list_relation_prompt_versions(db_path: Path) -> list[str]:
    """List prompt versions available in relation_llm_reviews."""
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT prompt_version
            FROM relation_llm_reviews
            WHERE COALESCE(TRIM(prompt_version), '') <> ''
            ORDER BY prompt_version DESC
            LIMIT 100
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def list_relation_types(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT relation_type
            FROM relation_extractions
            WHERE COALESCE(TRIM(relation_type), '') <> ''
            ORDER BY relation_type ASC
            LIMIT 100
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def list_relation_scopes(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT scope
            FROM relation_extractions
            WHERE COALESCE(TRIM(scope), '') <> ''
            ORDER BY scope ASC
            LIMIT 100
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def _relations_prompt_filter(prompt_version: str | None) -> tuple[str, list[Any]]:
    if not prompt_version:
        return "", []
    return (
        "WHERE relation_id IN ("
        "SELECT relation_id FROM relation_llm_reviews WHERE prompt_version = ?"
        ")",
        [prompt_version],
    )


def _group_count(
    conn: sqlite3.Connection,
    field_name: str,
    base_where: str,
    params: list[Any],
) -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT {field_name} AS label, COUNT(*) AS total
        FROM relation_extractions
        {base_where}
        GROUP BY {field_name}
        ORDER BY total DESC, label ASC
        LIMIT 50
        """,
        params,
    ).fetchall()
    return {str(row["label"]): int(row["total"]) for row in rows}
