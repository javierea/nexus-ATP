"""Streamlit UI for Etapa 4 (citas normativas)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rg_atp_pipeline.config import load_config
from rg_atp_pipeline.paths import config_path, data_dir
from rg_atp_pipeline.services.citations_service import run_citations
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def render_citations_stage(db_path: Path) -> None:
    st.title("Etapa 4 — Citas normativas")
    config = load_config(config_path())
    data_root = data_dir()

    llm_mode = st.radio(
        "Modo LLM",
        ["off", "verify", "verify_all"],
        index=0,
        horizontal=True,
        key="citations_llm_mode",
    )

    with st.form("citations_form"):
        st.subheader("Parámetros de ejecución")
        doc_keys_raw = st.text_input(
            "Doc keys (separados por coma o espacio)",
            help="Opcional. Ej: RG-2024-001, RG-2024-002",
        )
        limit_docs = st.text_input("Límite de documentos (opcional)")
        min_confidence = st.slider(
            "Confianza mínima (regex)",
            0.0,
            1.0,
            0.7,
            0.05,
        )
        create_placeholders = st.checkbox(
            "Crear placeholders en catálogo de normas",
            value=True,
        )

        ollama_base_url = config.ollama_base_url
        ollama_model = config.ollama_model
        batch_size = config.llm_batch_size
        prompt_version = config.llm_prompt_version
        llm_gate_regex_threshold = config.llm_gate_regex_threshold

        st.markdown("**Overrides Ollama (opcional)**")
        overrides_disabled = llm_mode not in {"verify", "verify_all"}
        ollama_base_url = st.text_input(
            "Base URL Ollama",
            value=config.ollama_base_url,
            disabled=overrides_disabled,
        )
        ollama_model = st.text_input(
            "Modelo Ollama",
            value=config.ollama_model,
            disabled=overrides_disabled,
        )
        batch_size = st.number_input(
            "Batch size LLM",
            min_value=1,
            value=config.llm_batch_size,
            step=1,
            disabled=overrides_disabled,
        )
        prompt_version = st.text_input(
            "Prompt version",
            value=config.llm_prompt_version,
            disabled=overrides_disabled,
        )
        extract_version = st.text_input(
            "Extract version",
            value="citext-v2",
        )
        llm_gate_regex_threshold = st.slider(
            "Threshold gating regex (LLM)",
            0.0,
            1.0,
            float(config.llm_gate_regex_threshold),
            0.01,
            disabled=overrides_disabled,
        )
        if llm_mode in {"verify", "verify_all"}:
            test_ollama = st.form_submit_button("Probar conexión Ollama")
        else:
            test_ollama = False

        submitted = st.form_submit_button("Ejecutar Etapa 4")

    if test_ollama:
        try:
            response = requests.get(
                _ollama_tags_endpoint(ollama_base_url),
                timeout=3,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.warning(f"Ollama no responde: {exc}")
        else:
            st.success("Ollama responde OK.")

    doc_keys = parse_doc_keys(doc_keys_raw)
    doc_keys = doc_keys or None
    limit_docs_value = _parse_optional_int(limit_docs)

    if submitted:
        progress_text = st.empty()
        progress_bar = st.progress(0.0)

        def _on_progress(payload: dict[str, Any]) -> None:
            stage = str(payload.get("stage") or "")
            current = int(payload.get("current") or 0)
            total = max(1, int(payload.get("total") or 1))
            message = str(payload.get("message") or "")
            if stage == "docs":
                ratio = (current / total) * 0.8
            elif stage == "llm":
                ratio = 0.8 + ((current / total) * 0.2)
            else:
                ratio = current / total
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            if message:
                progress_text.caption(message)

        try:
            summary = run_citations(
                db_path=db_path,
                data_dir=data_root,
                doc_keys=doc_keys,
                limit_docs=limit_docs_value,
                llm_mode=llm_mode,
                min_confidence=min_confidence,
                create_placeholders=create_placeholders,
                batch_size=int(batch_size) if llm_mode in {"verify", "verify_all"} else None,
                ollama_model=ollama_model if llm_mode in {"verify", "verify_all"} else None,
                ollama_base_url=ollama_base_url if llm_mode in {"verify", "verify_all"} else None,
                prompt_version=prompt_version,
                llm_gate_regex_threshold=llm_gate_regex_threshold,
                extract_version=extract_version,
                progress_callback=_on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - show runtime errors.
            progress_bar.empty()
            progress_text.empty()
            st.error(f"Error al ejecutar Etapa 4: {exc}")
        else:
            progress_bar.progress(1.0)
            progress_text.caption("Etapa 4 finalizada.")
            st.session_state["citations_summary"] = summary
            st.session_state["citations_doc_keys"] = doc_keys
            st.success("Etapa 4 completada.")
            st.cache_data.clear()

    summary = st.session_state.get("citations_summary")
    doc_keys = st.session_state.get("citations_doc_keys", doc_keys)
    if summary:
        if summary.get("docs_processed", 0) == 0:
            st.info("No se procesaron documentos. Verifique doc_keys o textos.")
        st.subheader("Resumen")
        st.json(summary)

    st.subheader("Preview SQLite")
    show_preview = st.checkbox("Mostrar preview", value=False)
    if show_preview:
        preview_limit = st.number_input(
            "Filas a mostrar",
            min_value=1,
            max_value=200,
            value=50,
            step=10,
        )
        citations_rows = _read_citations_preview(
            db_path,
            int(preview_limit),
            doc_keys,
            extract_version=extract_version,
        )
        st.caption("Tabla citations (recientes)")
        st.dataframe(_maybe_dataframe(citations_rows))

        links_rows = _read_citation_links_preview(db_path, int(preview_limit))
        st.caption("Tabla citation_links (recientes)")
        st.dataframe(_maybe_dataframe(links_rows))


def parse_doc_keys(value: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    return list(dict.fromkeys(items))


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_citations_preview(
    db_path: Path,
    limit: int,
    doc_keys: list[str] | None,
    extract_version: str | None = None,
) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    query = """
        SELECT
            c.citation_id,
            c.source_doc_key,
            c.source_unit_id,
            c.source_unit_type,
            u.unit_number AS evidence_unit_number,
            c.evidence_kind,
            c.extract_version,
            c.raw_text,
            c.norm_type_guess,
            c.norm_key_candidate,
            c.evidence_text,
            u.text AS evidence_unit_full_text,
            c.regex_confidence,
            c.detected_at
        FROM citations c
        LEFT JOIN units u ON u.id = c.evidence_unit_id
    """
    params: list[Any] = []
    if doc_keys:
        placeholders = ", ".join("?" for _ in doc_keys)
        query += f" WHERE c.source_doc_key IN ({placeholders})"
        params.extend(doc_keys)
    if extract_version:
        query += " AND" if " WHERE " in query else " WHERE "
        query += " c.extract_version = ?"
        params.append(extract_version)
    else:
        query += " AND" if " WHERE " in query else " WHERE "
        query += " c.extract_version = (SELECT extract_version FROM citations ORDER BY detected_at DESC, citation_id DESC LIMIT 1)"
    query += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _read_citation_links_preview(db_path: Path, limit: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    query = """
        SELECT
            link_id,
            citation_id,
            target_norm_id,
            target_norm_key,
            resolution_status,
            resolution_confidence,
            created_at
        FROM citation_links
        ORDER BY created_at DESC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (limit,)).fetchall()
    return [dict(row) for row in rows]


def _ollama_tags_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/api/generate"):
        endpoint = endpoint[: -len("/api/generate")]
    if endpoint.endswith("/api/chat"):
        endpoint = endpoint[: -len("/api/chat")]
    return f"{endpoint}/api/tags"


def _maybe_dataframe(records: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(records)
    except ImportError:
        return records


def get_citations_summary(db_path: Path) -> dict[str, Any]:
    """Return aggregated metrics for Stage 4 and pipeline overview cards."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        doc_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_rgs,
                SUM(CASE WHEN ds.structure_status = 'STRUCTURED' THEN 1 ELSE 0 END)
                    AS rgs_with_structure,
                SUM(
                    CASE
                        WHEN ds.structure_status = 'STRUCTURED'
                             AND COALESCE(d.text_status, 'NONE') = 'EXTRACTED'
                        THEN 1
                        ELSE 0
                    END
                ) AS rgs_ready_for_rag
            FROM documents d
            LEFT JOIN doc_structure ds ON ds.doc_key = d.doc_key
            """
        ).fetchone()
        citation_row = conn.execute(
            """
            WITH latest_review AS (
                SELECT llm_model, prompt_version, created_at
                FROM citation_llm_reviews
                ORDER BY created_at DESC, review_id DESC
                LIMIT 1
            )
            SELECT
                (SELECT COUNT(*) FROM citations) AS total_citations,
                (SELECT COUNT(*) FROM citation_llm_reviews) AS total_reviews,
                (SELECT COUNT(*) FROM citation_links WHERE resolution_status = 'RESOLVED')
                    AS total_resolved,
                (
                    SELECT COUNT(*)
                    FROM citation_links
                    WHERE resolution_status = 'PLACEHOLDER_CREATED'
                ) AS total_placeholder_created,
                (SELECT COUNT(*) FROM citation_links WHERE resolution_status = 'REJECTED')
                    AS total_rejected,
                (SELECT prompt_version FROM latest_review) AS last_prompt_version,
                (SELECT llm_model FROM latest_review) AS last_model,
                (SELECT created_at FROM latest_review) AS last_review_at
            """
        ).fetchone()
        versions = conn.execute(
            """
            SELECT prompt_version, COUNT(*) AS total_reviews
            FROM citation_llm_reviews
            GROUP BY prompt_version
            ORDER BY total_reviews DESC, prompt_version DESC
            """
        ).fetchall()

    return {
        "total_rgs": (doc_row["total_rgs"] or 0) if doc_row else 0,
        "rgs_with_structure": (doc_row["rgs_with_structure"] or 0) if doc_row else 0,
        "rgs_ready_for_rag": (doc_row["rgs_ready_for_rag"] or 0) if doc_row else 0,
        "total_citations": (citation_row["total_citations"] or 0) if citation_row else 0,
        "total_reviews": (citation_row["total_reviews"] or 0) if citation_row else 0,
        "total_resolved": (citation_row["total_resolved"] or 0) if citation_row else 0,
        "total_placeholder_created": (
            (citation_row["total_placeholder_created"] or 0) if citation_row else 0
        ),
        "total_rejected": (citation_row["total_rejected"] or 0) if citation_row else 0,
        "last_prompt_version": citation_row["last_prompt_version"] if citation_row else None,
        "last_model": citation_row["last_model"] if citation_row else None,
        "last_review_at": citation_row["last_review_at"] if citation_row else None,
        "reviews_by_prompt_version": [dict(row) for row in versions],
    }


def get_citations_breakdown(
    db_path: Path,
    prompt_version: str | None = None,
    resolution_status: str | None = None,
    norm_type: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return filtered citations rows for analysis table."""
    if not db_path.exists():
        return []
    query = """
        SELECT
            c.citation_id,
            c.raw_text,
            cl.resolution_status,
            cl.target_norm_key,
            lr.is_reference,
            lr.llm_confidence,
            lr.explanation,
            lr.prompt_version,
            lr.norm_type,
            lr.created_at
        FROM citations c
        LEFT JOIN citation_links cl ON cl.citation_id = c.citation_id
        LEFT JOIN citation_llm_reviews lr ON lr.citation_id = c.citation_id
    """
    where: list[str] = []
    params: list[Any] = []
    if prompt_version:
        where.append("lr.prompt_version = ?")
        params.append(prompt_version)
    if resolution_status:
        where.append("cl.resolution_status = ?")
        params.append(resolution_status)
    if norm_type:
        where.append("COALESCE(lr.norm_type, c.norm_type_guess) = ?")
        params.append(norm_type)
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY COALESCE(lr.created_at, cl.created_at, c.detected_at) DESC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_llm_explanations_stats(
    db_path: Path,
    prompt_version: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return top LLM explanations by frequency."""
    if not db_path.exists():
        return []
    query = """
        SELECT explanation, COUNT(*) AS total
        FROM citation_llm_reviews
    """
    params: list[Any] = []
    if prompt_version:
        query += " WHERE prompt_version = ?"
        params.append(prompt_version)
    query += " GROUP BY explanation ORDER BY total DESC, explanation ASC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_citation_filter_options(db_path: Path) -> dict[str, list[str]]:
    """Return available filter values for citations analysis."""
    if not db_path.exists():
        return {"prompt_versions": [], "resolution_statuses": [], "norm_types": []}
    with sqlite3.connect(db_path) as conn:
        prompt_versions = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT prompt_version FROM citation_llm_reviews ORDER BY prompt_version DESC"
            ).fetchall()
            if row[0]
        ]
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT resolution_status FROM citation_links ORDER BY resolution_status"
            ).fetchall()
            if row[0]
        ]
        norm_types = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT norm_type
                FROM citation_llm_reviews
                WHERE norm_type IS NOT NULL
                ORDER BY norm_type
                """
            ).fetchall()
            if row[0]
        ]
    return {
        "prompt_versions": prompt_versions,
        "resolution_statuses": statuses,
        "norm_types": norm_types,
    }


def get_norms_coverage_stats(db_path: Path, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    """Return aggregated norms coverage data for dashboard tab."""
    if not db_path.exists():
        return {
            "top_cited_norms": [],
            "norms_with_placeholders": [],
            "most_used_aliases": [],
            "cited_never_resolved": [],
        }
    repo = NormsRepository(db_path=db_path)
    with repo._connection() as conn:  # noqa: SLF001 - repository-managed connection reuse.
        top_cited_norms = conn.execute(
            """
            SELECT
                cl.target_norm_key,
                COUNT(*) AS total_citations
            FROM citation_links cl
            WHERE cl.target_norm_key IS NOT NULL
            GROUP BY cl.target_norm_key
            ORDER BY total_citations DESC, cl.target_norm_key ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        norms_with_placeholders = conn.execute(
            """
            SELECT
                cl.target_norm_key,
                COUNT(*) AS total_placeholders
            FROM citation_links cl
            WHERE cl.resolution_status = 'PLACEHOLDER_CREATED'
              AND cl.target_norm_key IS NOT NULL
            GROUP BY cl.target_norm_key
            ORDER BY total_placeholders DESC, cl.target_norm_key ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        most_used_aliases = conn.execute(
            """
            SELECT
                na.alias_text,
                n.norm_key,
                COUNT(*) AS uses
            FROM citation_llm_reviews lr
            JOIN norms n ON n.norm_key = lr.normalized_key
            JOIN norm_aliases na ON na.norm_id = n.norm_id
            GROUP BY na.alias_text, n.norm_key
            ORDER BY uses DESC, na.alias_text ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        cited_never_resolved = conn.execute(
            """
            SELECT
                COALESCE(cl.target_norm_key, c.norm_key_candidate) AS norm_key,
                COUNT(*) AS mentions
            FROM citations c
            LEFT JOIN citation_links cl ON cl.citation_id = c.citation_id
            GROUP BY COALESCE(cl.target_norm_key, c.norm_key_candidate)
            HAVING norm_key IS NOT NULL
               AND SUM(CASE WHEN cl.resolution_status = 'RESOLVED' THEN 1 ELSE 0 END) = 0
            ORDER BY mentions DESC, norm_key ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "top_cited_norms": [dict(row) for row in top_cited_norms],
        "norms_with_placeholders": [dict(row) for row in norms_with_placeholders],
        "most_used_aliases": [dict(row) for row in most_used_aliases],
        "cited_never_resolved": [dict(row) for row in cited_never_resolved],
    }


def get_consistency_issues(db_path: Path) -> dict[str, Any]:
    """Return inconsistencies between LLM flag, explanation and link resolution."""
    if not db_path.exists():
        return {"total_inconsistencies": 0, "details": []}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                c.citation_id,
                c.raw_text,
                lr.is_reference,
                lr.explanation,
                cl.resolution_status,
                CASE
                    WHEN lr.is_reference = 1 AND lower(lr.explanation) LIKE '%no es%'
                    THEN 'is_reference=1 + explanation contiene "No es"'
                    WHEN lr.is_reference = 0 AND cl.resolution_status = 'RESOLVED'
                    THEN 'is_reference=0 + resolution_status=RESOLVED'
                    ELSE 'other'
                END AS issue_type
            FROM citation_llm_reviews lr
            JOIN citations c ON c.citation_id = lr.citation_id
            LEFT JOIN citation_links cl ON cl.citation_id = c.citation_id
            WHERE (lr.is_reference = 1 AND lower(lr.explanation) LIKE '%no es%')
               OR (lr.is_reference = 0 AND cl.resolution_status = 'RESOLVED')
            ORDER BY lr.created_at DESC
            LIMIT 200
            """
        ).fetchall()
    return {
        "total_inconsistencies": len(rows),
        "details": [dict(row) for row in rows],
    }
