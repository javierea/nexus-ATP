"""Streamlit control panel for rg_atp_pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import streamlit as st

from rg_atp_pipeline.cli import validate_config_state
from rg_atp_pipeline.config import Config, load_config, save_config
from rg_atp_pipeline.fetcher import FetchOptions, run_fetch
from rg_atp_pipeline.logging_utils import setup_logging
from rg_atp_pipeline.paths import config_path, data_dir, state_path
from rg_atp_pipeline.project import init_project
from rg_atp_pipeline.queries import (
    get_confidence_distribution,
    get_filter_options,
    get_kpis,
    list_backlog,
    list_documents,
    recent_activity,
)
from rg_atp_pipeline.state import load_state
from rg_atp_pipeline.storage_sqlite import DocumentStore
from rg_atp_pipeline.structure_segmenter import StructureOptions, run_structure
from rg_atp_pipeline.text_extractor import ExtractOptions, run_extract
from rg_atp_pipeline.audit_compendio import (
    review_missing_downloads,
    run_audit_compendio,
    update_audit_summary_with_review,
)
from rg_atp_pipeline.citations_ui import render_citations_stage
from rg_atp_pipeline.citations_ui import (
    get_citation_filter_options,
    get_citations_breakdown,
    get_citations_summary,
    get_consistency_issues,
    get_llm_explanations_stats,
    get_norms_coverage_stats,
)
from rg_atp_pipeline.ollama_client import (
    OllamaClient,
    OllamaConfig,
    OllamaReviewer,
    OllamaUnavailableError,
)
from rg_atp_pipeline.norms_ui import (
    resolve_norm_ui,
    seed_catalog,
    upload_norm_pdf_ui,
)
from rg_atp_pipeline.relations_ui import (
    get_relations_inconsistencies,
    get_relations_qa_samples,
    get_relations_summary,
    get_relations_table,
    list_relation_prompt_versions,
    list_relation_scopes,
    list_relation_types,
    run_relations_ui,
)


st.set_page_config(page_title="RG ATP Control Panel", layout="wide")


@st.cache_data(ttl=10)
def cached_kpis(db_path_str: str) -> dict[str, int]:
    return get_kpis(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_confidence(db_path_str: str) -> dict[str, int]:
    return get_confidence_distribution(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_filters(db_path_str: str) -> dict[str, list[Any]]:
    return get_filter_options(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_documents(db_path_str: str, filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return list_documents(Path(db_path_str), filters, limit=limit)


@st.cache_data(ttl=10)
def cached_backlog(db_path_str: str, kind: str, limit: int) -> list[dict[str, Any]]:
    return list_backlog(Path(db_path_str), kind=kind, limit=limit)


@st.cache_data(ttl=10)
def cached_activity(db_path_str: str, limit: int) -> list[dict[str, Any]]:
    return recent_activity(Path(db_path_str), limit=limit)


@st.cache_data(ttl=10)
def cached_log_tail(path_str: str, lines: int) -> str:
    return tail_file(Path(path_str), lines)


def maybe_dataframe(records: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(records)
    except ImportError:
        return records


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def parse_doc_keys_input(value: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    return list(dict.fromkeys(items))


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_years(value: str) -> list[int]:
    items = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [int(item) for item in items]


def parse_max_n_by_year(value: str) -> dict[str, int]:
    if not value.strip():
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("max_n_by_year debe ser un JSON con claves y valores.")
    return {str(key): int(val) for key, val in payload.items()}


def tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return "(archivo no encontrado)"
    data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(data[-lines:])


def run_app() -> None:
    init_project()
    logger = setup_logging(data_dir() / "logs")
    db_path = data_dir() / "state" / "rg_atp.sqlite"
    store = DocumentStore(db_path)

    page = st.sidebar.radio(
        "Módulo",
        [
            "Pipeline Overview",
            "Fetch",
            "Extract",
            "Structure",
            "Audit",
            "Etapa 4 — Citas",
            "Etapa 4.1 — Relaciones",
            "Normas",
            "Config",
            "Logs",
        ],
    )

    if page == "Pipeline Overview":
        render_pipeline_overview(db_path)
    elif page == "Fetch":
        render_fetch(db_path, store, logger)
    elif page == "Extract":
        render_extract(db_path, store, logger)
    elif page == "Structure":
        render_structure(db_path, store, logger)
    elif page == "Audit":
        render_audit(db_path)
    elif page == "Etapa 4 — Citas":
        render_citations_stage(db_path)
    elif page == "Etapa 4.1 — Relaciones":
        render_relations_stage(db_path)
    elif page == "Normas":
        render_normas(db_path)
    elif page == "Config":
        render_config()
    elif page == "Logs":
        render_logs()



def render_dashboard(db_path: Path) -> None:
    st.title("Dashboard")
    kpis = cached_kpis(str(db_path))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total docs", kpis.get("total_docs", 0))
    col2.metric("Descargados", kpis.get("downloaded", 0))
    col3.metric("Missing", kpis.get("missing", 0))
    col4.metric("Errores fetch", kpis.get("error", 0))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Texto extraído", kpis.get("text_extracted", 0))
    col6.metric("NEEDS_OCR", kpis.get("needs_ocr", 0))
    col7.metric("Errores texto", kpis.get("text_error", 0))
    col8.metric("Structured", kpis.get("structured_ok", 0))

    col9, col10, _ = st.columns(3)
    col9.metric("Partial", kpis.get("partial", 0))
    col10.metric("Errores estructura", kpis.get("structure_error", 0))

    st.subheader("Confianza de estructura")
    confidence = cached_confidence(str(db_path))
    confidence_table = [
        {"bucket": ">=0.8", "count": confidence.get("high", 0)},
        {"bucket": "0.6-0.79", "count": confidence.get("mid", 0)},
        {"bucket": "<0.6", "count": confidence.get("low", 0)},
    ]
    st.bar_chart(maybe_dataframe(confidence_table), x="bucket", y="count")

    st.subheader("Backlogs")
    backlog_cols = st.columns(3)
    with backlog_cols[0]:
        st.caption("NEEDS_OCR")
        st.dataframe(maybe_dataframe(cached_backlog(str(db_path), "needs_ocr", 50)))
    with backlog_cols[1]:
        st.caption("PARTIAL o baja confianza")
        st.dataframe(maybe_dataframe(cached_backlog(str(db_path), "partial_or_low_confidence", 50)))
    with backlog_cols[2]:
        st.caption("Errores")
        st.dataframe(maybe_dataframe(cached_backlog(str(db_path), "errors", 50)))

    st.subheader("Actividad reciente")
    st.dataframe(maybe_dataframe(cached_activity(str(db_path), 50)))


@st.cache_data(ttl=10)
def cached_citations_summary(db_path_str: str) -> dict[str, Any]:
    return get_citations_summary(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_citations_breakdown(
    db_path_str: str,
    prompt_version: str | None,
    resolution_status: str | None,
    norm_type: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    return get_citations_breakdown(
        Path(db_path_str),
        prompt_version=prompt_version,
        resolution_status=resolution_status,
        norm_type=norm_type,
        limit=limit,
    )


@st.cache_data(ttl=10)
def cached_llm_explanations(db_path_str: str, prompt_version: str | None) -> list[dict[str, Any]]:
    return get_llm_explanations_stats(Path(db_path_str), prompt_version=prompt_version)


@st.cache_data(ttl=10)
def cached_citation_filter_options(db_path_str: str) -> dict[str, list[str]]:
    return get_citation_filter_options(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_norms_coverage(db_path_str: str) -> dict[str, list[dict[str, Any]]]:
    return get_norms_coverage_stats(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_consistency_issues(db_path_str: str) -> dict[str, Any]:
    return get_consistency_issues(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_relations_summary(db_path_str: str, prompt_version: str | None = None) -> dict[str, Any]:
    return get_relations_summary(Path(db_path_str), prompt_version=prompt_version)


@st.cache_data(ttl=10)
def cached_relations_table(
    db_path_str: str,
    prompt_version: str | None,
    relation_type: str | None,
    scope: str | None,
    limit: int,
):
    return get_relations_table(
        Path(db_path_str),
        prompt_version=prompt_version,
        relation_type=relation_type,
        scope=scope,
        limit=limit,
    )


@st.cache_data(ttl=10)
def cached_relations_samples(db_path_str: str, relation_type: str, n: int):
    return get_relations_qa_samples(Path(db_path_str), relation_type=relation_type, n=n)


@st.cache_data(ttl=10)
def cached_relations_inconsistencies(db_path_str: str) -> dict[str, Any]:
    return get_relations_inconsistencies(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_relation_prompt_versions(db_path_str: str) -> list[str]:
    return list_relation_prompt_versions(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_relation_types(db_path_str: str) -> list[str]:
    return list_relation_types(Path(db_path_str))


@st.cache_data(ttl=10)
def cached_relation_scopes(db_path_str: str) -> list[str]:
    return list_relation_scopes(Path(db_path_str))


def render_pipeline_overview(db_path: Path) -> None:
    st.title("Pipeline Overview")
    summary = cached_citations_summary(str(db_path))

    docs_col1, docs_col2, docs_col3 = st.columns(3)
    docs_col1.metric("Total RGs", summary.get("total_rgs", 0))
    docs_col2.metric("RG con estructura", summary.get("rgs_with_structure", 0))
    docs_col3.metric("RG listas para RAG", summary.get("rgs_ready_for_rag", 0))

    st.subheader("Citations (Etapa 4)")
    cit_col1, cit_col2, cit_col3, cit_col4 = st.columns(4)
    cit_col1.metric("Total citations", summary.get("total_citations", 0))
    cit_col2.metric("Total reviews", summary.get("total_reviews", 0))
    cit_col3.metric("Total RESOLVED", summary.get("total_resolved", 0))
    cit_col4.metric("Total PLACEHOLDER_CREATED", summary.get("total_placeholder_created", 0))

    cit_col5, cit_col6, cit_col7 = st.columns(3)
    cit_col5.metric("Total REJECTED", summary.get("total_rejected", 0))
    cit_col6.metric("Último prompt_version", summary.get("last_prompt_version") or "N/A")
    cit_col7.metric("Último modelo", summary.get("last_model") or "N/A")
    st.metric("Última fecha de review", summary.get("last_review_at") or "N/A")

    st.caption("Reviews por prompt_version")
    st.dataframe(maybe_dataframe(summary.get("reviews_by_prompt_version", [])))

    st.subheader("Relations (Etapa 4.1)")
    rel_summary = cached_relations_summary(str(db_path))
    rel_col1, rel_col2, rel_col3, rel_col4 = st.columns(4)
    rel_col1.metric("Total relations", rel_summary.get("total_relations", 0))
    rel_col2.metric("Docs covered", rel_summary.get("docs_covered", 0))
    rel_col3.metric("LLM reviews", rel_summary.get("llm_reviews_count", 0))
    rel_col4.metric("Última extracción", rel_summary.get("last_created_at") or "N/A")

    rel_top_types = [
        {"relation_type": relation_type, "count": count}
        for relation_type, count in list((rel_summary.get("by_type") or {}).items())[:5]
    ]
    st.caption("Top relation types")
    if rel_top_types:
        st.bar_chart(maybe_dataframe(rel_top_types), x="relation_type", y="count")
    else:
        st.info("Sin relaciones registradas.")

    st.subheader("Deterministic vs LLM")
    latest_run = st.session_state.get("citations_summary", {})
    det_col1, det_col2, det_col3 = st.columns(3)
    det_col1.metric(
        "resolved_deterministic_now",
        latest_run.get("resolved_deterministic_now", "N/A"),
    )
    det_col2.metric(
        "rejected_by_llm_now",
        latest_run.get("rejected_by_llm_now", "N/A"),
    )
    det_col3.metric(
        "llm_overruled_by_deterministic_now",
        latest_run.get("llm_overruled_by_deterministic_now", "N/A"),
    )

    consistency = cached_consistency_issues(str(db_path))
    total_inconsistencies = consistency.get("total_inconsistencies", 0)
    st.metric("Incoherencias", total_inconsistencies)
    if total_inconsistencies > 0:
        st.error(f"Se detectaron {total_inconsistencies} incoherencias de consistencia.")
    else:
        st.success("Sin incoherencias detectadas.")
    with st.expander("Detalles de incoherencias (JSON)"):
        st.json(consistency)

    analysis_tab, coverage_tab = st.tabs(["Citations Analysis", "Norms Coverage"])

    with analysis_tab:
        filter_options = cached_citation_filter_options(str(db_path))
        selected_prompt = st.selectbox(
            "Filtro prompt_version",
            options=["(todos)", *filter_options.get("prompt_versions", [])],
            index=0,
        )
        selected_status = st.selectbox(
            "Filtro resolution_status",
            options=["(todos)", *filter_options.get("resolution_statuses", [])],
            index=0,
        )
        selected_norm_type = st.selectbox(
            "Filtro norm_type",
            options=["(todos)", *filter_options.get("norm_types", [])],
            index=0,
        )
        prompt_filter = None if selected_prompt == "(todos)" else selected_prompt
        status_filter = None if selected_status == "(todos)" else selected_status
        norm_filter = None if selected_norm_type == "(todos)" else selected_norm_type

        st.caption("Top 20 explicaciones LLM más frecuentes")
        explanations = cached_llm_explanations(str(db_path), prompt_filter)
        st.dataframe(maybe_dataframe(explanations))

        rows = cached_citations_breakdown(
            str(db_path),
            prompt_filter,
            status_filter,
            norm_filter,
            200,
        )
        st.caption("Detalle de citas")
        st.dataframe(
            maybe_dataframe(
                [
                    {
                        "citation_id": row.get("citation_id"),
                        "raw_text": row.get("raw_text"),
                        "resolution_status": row.get("resolution_status"),
                        "target_norm_key": row.get("target_norm_key"),
                        "is_reference": row.get("is_reference"),
                        "llm_confidence": row.get("llm_confidence"),
                        "explanation": row.get("explanation"),
                    }
                    for row in rows
                ]
            )
        )
        with st.expander("Detalles JSON"):
            st.json(rows)

    with coverage_tab:
        coverage = cached_norms_coverage(str(db_path))
        st.caption("Normas más citadas (top 20)")
        st.dataframe(maybe_dataframe(coverage.get("top_cited_norms", [])))

        st.caption("Normas con más placeholders")
        st.dataframe(maybe_dataframe(coverage.get("norms_with_placeholders", [])))

        st.caption("Alias más utilizados")
        st.dataframe(maybe_dataframe(coverage.get("most_used_aliases", [])))

        st.caption("Normas citadas pero nunca resueltas")
        st.dataframe(maybe_dataframe(coverage.get("cited_never_resolved", [])))

        with st.expander("Cobertura (JSON)"):
            st.json(coverage)



def render_relations_stage(db_path: Path) -> None:
    st.title("Etapa 4.1 — Relaciones")
    config = load_config(config_path())
    data_root = data_dir()

    run_tab, summary_tab, explore_tab, audit_tab = st.tabs(["Ejecutar 4.1", "Resumen", "Explorar", "Auditoría"])

    with run_tab:
        with st.form("relations_form"):
            doc_keys_raw = st.text_input("Doc keys (coma o espacio)")
            limit_docs_raw = st.text_input("Límite de documentos (opcional)")
            llm_mode = st.selectbox("Modo LLM", ["off", "verify"], index=0)
            min_confidence = st.slider("Confianza mínima", 0.5, 0.95, 0.9, 0.01)
            prompt_version = st.text_input("Prompt version", value="reltype-v1")

            default_batch_size = int(getattr(config, "llm_batch_size", 20) or 20)
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                value=default_batch_size,
                step=1
            )
            ollama_model = st.text_input(
                "Modelo Ollama (opcional)",
                value=config.ollama_model
            )
            ollama_base_url = st.text_input(
                "Base URL Ollama (opcional)",
                value=config.ollama_base_url
            )
            submitted = st.form_submit_button("Correr Etapa 4.1")

        if submitted:
            doc_keys = parse_doc_keys_input(doc_keys_raw) or None
            limit_docs = parse_optional_int(limit_docs_raw)
            try:
                with st.spinner("Ejecutando Etapa 4.1..."):
                    summary = run_relations_ui(
                        db_path=db_path,
                        data_dir=data_root,
                        doc_keys=doc_keys,
                        limit_docs=limit_docs,
                        llm_mode=llm_mode,
                        min_confidence=min_confidence,
                        prompt_version=prompt_version,
                        batch_size=int(batch_size),
                        ollama_model=ollama_model if llm_mode == "verify" else None,
                        ollama_base_url=ollama_base_url if llm_mode == "verify" else None,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error al ejecutar Etapa 4.1: {exc}")
            else:
                st.session_state["relations_summary"] = summary
                st.success("Etapa 4.1 completada.")
                st.json(summary)
                col1, col2, col3 = st.columns(3)
                col1.metric("relations_inserted", summary.get("relations_inserted", 0))
                col2.metric("links_seen", summary.get("links_seen", 0))
                col3.metric("llm_verified", summary.get("llm_verified", 0))
                col4, col5, col6 = st.columns(3)
                col4.metric("gated_count", summary.get("gated_count", 0))
                col5.metric("batches_sent", summary.get("batches_sent", 0))
                col6.metric(
                    "skipped_already_reviewed_count",
                    summary.get("skipped_already_reviewed_count", 0),
                )
                col7, col8, _ = st.columns(3)
                col7.metric(
                    "skipped_according_to_no_target_now",
                    summary.get("skipped_according_to_no_target_now", 0),
                )
                col8.metric(
                    "inserted_according_to_with_target_now",
                    summary.get("inserted_according_to_with_target_now", 0),
                )
                if summary.get("llm_mode_effective") == "verify" and int(summary.get("gated_count", 0) or 0) == 0:
                    st.warning("LLM en modo verify sin candidatos gated (gated_count=0).")
                st.cache_data.clear()

    with summary_tab:
        summary = cached_relations_summary(str(db_path))
        latest_run = st.session_state.get("relations_summary") or {}
        col1, col2, col3 = st.columns(3)
        col1.metric("Total relations", summary.get("total_relations", 0))
        col2.metric("Docs covered", summary.get("docs_covered", 0))
        col3.metric("LLM reviews", summary.get("llm_reviews_count", 0))
        col4, col5, _ = st.columns(3)
        col4.metric("Última extracción", summary.get("last_created_at") or "N/A")
        col5.metric("Último modelo LLM", summary.get("last_llm_model") or "N/A")
        st.metric("Último prompt", summary.get("last_prompt_version") or "N/A")

        run_col1, run_col2, run_col3 = st.columns(3)
        run_col1.metric("gated_count", latest_run.get("gated_count", 0))
        run_col2.metric("batches_sent", latest_run.get("batches_sent", 0))
        run_col3.metric(
            "skipped_already_reviewed_count",
            latest_run.get("skipped_already_reviewed_count", 0),
        )
        run_col4, run_col5, _ = st.columns(3)
        run_col4.metric(
            "skipped_according_to_no_target_now",
            latest_run.get("skipped_according_to_no_target_now", 0),
        )
        run_col5.metric(
            "inserted_according_to_with_target_now",
            latest_run.get("inserted_according_to_with_target_now", 0),
        )
        if latest_run.get("llm_mode_effective") == "verify" and int(latest_run.get("gated_count", 0) or 0) == 0:
            st.warning("Última corrida en modo verify sin candidatos gated (gated_count=0).")

        by_type = [
            {"relation_type": key, "count": value}
            for key, value in (summary.get("by_type") or {}).items()
        ]
        by_scope = [{"scope": key, "count": value} for key, value in (summary.get("by_scope") or {}).items()]
        st.caption("Distribución por tipo")
        if by_type:
            st.bar_chart(maybe_dataframe(by_type), x="relation_type", y="count")
        else:
            st.info("No hay datos por tipo.")
        st.caption("Distribución por alcance")
        if by_scope:
            st.bar_chart(maybe_dataframe(by_scope), x="scope", y="count")
        else:
            st.info("No hay datos por alcance.")

    with explore_tab:
        prompt_versions = cached_relation_prompt_versions(str(db_path))
        relation_types = cached_relation_types(str(db_path))
        scopes = cached_relation_scopes(str(db_path))

        col1, col2, col3, col4 = st.columns(4)
        selected_prompt = col1.selectbox("prompt_version", ["Todos", *prompt_versions], index=0)
        selected_type = col2.selectbox("relation_type", ["Todos", *relation_types], index=0)
        selected_scope = col3.selectbox("scope", ["Todos", *scopes], index=0)
        limit_rows = col4.slider("Límite", min_value=100, max_value=2000, value=500, step=100)

        if selected_prompt != "Todos":
            st.caption(
                "Se muestran todas las relaciones; las columnas LLM corresponden a "
                "prompt_version seleccionado (si existe review)."
            )

        table = cached_relations_table(
            str(db_path),
            None if selected_prompt == "Todos" else selected_prompt,
            None if selected_type == "Todos" else selected_type,
            None if selected_scope == "Todos" else selected_scope,
            limit_rows,
        )

        if hasattr(table, "empty"):
            has_rows = not table.empty
        else:
            has_rows = bool(table)

        if has_rows:
            if hasattr(table, "copy"):
                preview = table.copy()
                preview["evidence_snippet"] = preview["evidence_snippet"].astype(str).str.slice(0, 120)
                preview["explanation"] = preview["explanation"].astype(str).str.slice(0, 120)
                st.dataframe(preview, use_container_width=True)

                selected_idx = st.number_input("Fila para detalle", min_value=0, max_value=len(table) - 1, value=0, step=1)
                detail = table.iloc[int(selected_idx)].to_dict()
            else:
                preview = [
                    {
                        **row,
                        "evidence_snippet": str(row.get("evidence_snippet", ""))[:120],
                        "explanation": str(row.get("explanation", ""))[:120],
                    }
                    for row in table
                ]
                st.dataframe(preview, use_container_width=True)
                selected_idx = st.number_input("Fila para detalle", min_value=0, max_value=len(table) - 1, value=0, step=1)
                detail = table[int(selected_idx)]
            with st.expander("Detalle de registro"):
                st.json(detail)
        else:
            st.info("Sin resultados con los filtros actuales.")

    with audit_tab:
        relation_type = st.selectbox(
            "relation_type para muestreo",
            ["REPEALS", "MODIFIES", "SUBSTITUTES", "ACCORDING_TO", "UNKNOWN"],
            index=0,
        )
        sample_size = st.slider("Tamaño de muestra", min_value=5, max_value=100, value=30, step=5)
        if st.button("Muestrear"):
            samples = cached_relations_samples(str(db_path), relation_type, sample_size)
            st.dataframe(samples, use_container_width=True)

        inconsistencies = cached_relations_inconsistencies(str(db_path))
        st.subheader("Inconsistencias")
        st.json(inconsistencies)

        if inconsistencies.get("count_article_without_detail", 0) > 0:
            st.warning("Hay relaciones ARTICLE sin scope_detail.")
        if inconsistencies.get("count_unknown_high_conf", 0) > 0:
            st.warning("Hay UNKNOWN con alta confianza (>=0.8).")
        if inconsistencies.get("count_effect_without_target", 0) > 0:
            st.warning("Hay relaciones sin target_norm_key.")

        latest_run = st.session_state.get("relations_summary") or {}
        st.subheader("Descartes ACCORDING_TO sin target")
        st.metric(
            "skipped_according_to_no_target_now",
            latest_run.get("skipped_according_to_no_target_now", 0),
        )
        skipped_samples = latest_run.get("skipped_according_to_no_target_samples") or []
        if skipped_samples:
            sample_limit = st.slider(
                "Muestra descartes ACCORDING_TO sin target",
                min_value=1,
                max_value=min(50, len(skipped_samples)),
                value=min(10, len(skipped_samples)),
                step=1,
            )
            st.dataframe(skipped_samples[:sample_limit], use_container_width=True)
        else:
            st.caption("No hay descartes ACCORDING_TO sin target en la última corrida.")




def render_fetch(db_path: Path, store: DocumentStore, logger: logging.Logger) -> None:
    st.title("Fetch")
    config = load_config(config_path())
    state = load_state(state_path())

    with st.form("fetch_form"):
        mode = st.selectbox("Modo", ["both", "new", "old"])
        year = st.text_input("Año (solo new)")
        n_start = st.text_input("N inicio (new)")
        n_end = st.text_input("N fin (new)")
        old_start = st.text_input("Old inicio")
        old_end = st.text_input("Old fin")
        max_downloads = st.text_input("Máximo descargas")
        dry_run = st.checkbox("Dry run")
        skip_existing = st.checkbox("Omitir existentes")
        submitted = st.form_submit_button("Ejecutar fetch")

    if submitted:
        options = FetchOptions(
            mode=mode,
            year=parse_optional_int(year),
            n_start=parse_optional_int(n_start),
            n_end=parse_optional_int(n_end),
            old_start=parse_optional_int(old_start),
            old_end=parse_optional_int(old_end),
            dry_run=dry_run,
            max_downloads=parse_optional_int(max_downloads),
            skip_existing=skip_existing,
        )
        with st.spinner("Ejecutando fetch..."):
            summary = run_fetch(config, state, store, data_dir(), options, logger)
        st.success(f"Fetch completado: {summary.as_dict()}")
        st.cache_data.clear()

    st.subheader("Documentos")
    filters = cached_filters(str(db_path))
    filter_cols = st.columns(3)
    with filter_cols[0]:
        status = st.selectbox("Status", ["Todos"] + filters["statuses"])
    with filter_cols[1]:
        family = st.selectbox("Familia", ["Todos"] + filters["doc_families"])
    with filter_cols[2]:
        year_filter = st.selectbox("Año", ["Todos"] + [str(y) for y in filters["years"]])

    list_filters: dict[str, Any] = {}
    if status != "Todos":
        list_filters["status"] = status
    if family != "Todos":
        list_filters["doc_family"] = family
    if year_filter != "Todos":
        list_filters["year"] = int(year_filter)

    docs = cached_documents(str(db_path), list_filters, limit=200)
    st.dataframe(maybe_dataframe(docs))



def render_extract(db_path: Path, store: DocumentStore, logger: logging.Logger) -> None:
    st.title("Extract")
    config = load_config(config_path())

    with st.form("extract_form"):
        doc_key = st.text_input("Doc key (opcional)")
        status = st.text_input("Status", value="DOWNLOADED")
        limit = st.text_input("Límite")
        force = st.checkbox("Reprocesar")
        only_needs_ocr = st.checkbox("Solo NEEDS_OCR")
        submitted = st.form_submit_button("Ejecutar extract")

    if submitted:
        options = ExtractOptions(
            status=status,
            limit=parse_optional_int(limit),
            doc_key=doc_key.strip() or None,
            force=force,
            only_text=False,
            only_needs_ocr=only_needs_ocr,
        )
        with st.spinner("Ejecutando extracción..."):
            summary = run_extract(config, store, data_dir(), options, logger)
        st.success(f"Extracción completada: {summary.as_dict()}")
        st.cache_data.clear()

    st.subheader("Documentos")
    filters = cached_filters(str(db_path))
    filter_cols = st.columns(3)
    with filter_cols[0]:
        text_status = st.selectbox("Text status", ["Todos"] + filters["text_statuses"])
    with filter_cols[1]:
        family = st.selectbox("Familia", ["Todos"] + filters["doc_families"])
    with filter_cols[2]:
        year_filter = st.selectbox("Año", ["Todos"] + [str(y) for y in filters["years"]])

    list_filters: dict[str, Any] = {}
    if text_status != "Todos":
        list_filters["text_status"] = text_status
    if family != "Todos":
        list_filters["doc_family"] = family
    if year_filter != "Todos":
        list_filters["year"] = int(year_filter)

    docs = cached_documents(str(db_path), list_filters, limit=200)
    columns = [
        "doc_key",
        "doc_family",
        "year",
        "text_status",
        "char_count",
        "pages_total",
        "pages_with_text",
        "alpha_ratio",
        "text_extracted_at",
    ]
    filtered = [{key: doc.get(key) for key in columns} for doc in docs]
    st.dataframe(maybe_dataframe(filtered))



def render_structure(db_path: Path, store: DocumentStore, logger: logging.Logger) -> None:
    st.title("Structure")

    with st.form("structure_form"):
        doc_key = st.text_input("Doc key (opcional)")
        limit = st.text_input("Límite")
        force = st.checkbox("Reprocesar")
        include_needs_ocr = st.checkbox("Incluir NEEDS_OCR")
        export_json = st.checkbox("Exportar JSON", value=True)
        submitted = st.form_submit_button("Ejecutar structure")

    if submitted:
        options = StructureOptions(
            doc_key=doc_key.strip() or None,
            limit=parse_optional_int(limit),
            force=force,
            include_needs_ocr=include_needs_ocr,
            export_json=export_json,
        )
        with st.spinner("Ejecutando structure..."):
            summary = run_structure(store, data_dir(), options, logger)
        st.success(f"Structure completado: {summary.as_dict()}")
        st.cache_data.clear()

    st.subheader("Documentos")
    filters = cached_filters(str(db_path))
    filter_cols = st.columns(3)
    with filter_cols[0]:
        structure_status = st.selectbox(
            "Structure status", ["Todos"] + filters["structure_statuses"]
        )
    with filter_cols[1]:
        family = st.selectbox("Familia", ["Todos"] + filters["doc_families"])
    with filter_cols[2]:
        year_filter = st.selectbox("Año", ["Todos"] + [str(y) for y in filters["years"]])

    list_filters: dict[str, Any] = {}
    if structure_status != "Todos":
        list_filters["structure_status"] = structure_status
    if family != "Todos":
        list_filters["doc_family"] = family
    if year_filter != "Todos":
        list_filters["year"] = int(year_filter)

    docs = cached_documents(str(db_path), list_filters, limit=200)
    columns = [
        "doc_key",
        "doc_family",
        "year",
        "structure_status",
        "articles_detected",
        "structure_confidence",
        "structured_at",
    ]
    filtered = [{key: doc.get(key) for key in columns} for doc in docs]
    st.dataframe(maybe_dataframe(filtered))

    st.subheader("Ver JSON estructurado")
    json_doc_key = st.selectbox(
        "Doc key", ["-"] + [doc["doc_key"] for doc in filtered if doc.get("doc_key")]
    )
    if json_doc_key and json_doc_key != "-":
        json_path = data_dir() / "structured" / f"{json_doc_key}.json"
        if json_path.exists():
            with st.expander("JSON"):
                st.code(json_path.read_text(encoding="utf-8"), language="json")
        else:
            st.info("No se encontró el JSON para este doc_key.")


def render_audit(db_path: Path) -> None:
    st.title("Audit")
    default_pdf = data_dir() / "compendio-legislativo-al-31-12-2024.pdf"

    with st.form("audit_form"):
        pdf_path = st.text_input("Ruta al PDF", value=str(default_pdf))
        uploaded = st.file_uploader("O cargar PDF", type=["pdf"])
        export_dir = st.text_input("Directorio export", value=str(data_dir() / "audit"))
        min_confidence = st.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.05)
        save_to_db = st.checkbox("Guardar histórico en SQLite", value=True)
        submitted = st.form_submit_button("Run")

    if submitted:
        if uploaded is not None:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            temp_path = export_path / uploaded.name
            temp_path.write_bytes(uploaded.getbuffer())
            pdf_path = str(temp_path)

        refs, summary = run_audit_compendio(
            Path(pdf_path),
            db_path,
            Path(export_dir),
            min_confidence=min_confidence,
            save_to_db=save_to_db,
        )
        st.session_state["audit_refs"] = refs
        st.session_state["audit_summary"] = summary
    else:
        refs = st.session_state.get("audit_refs")
        summary = st.session_state.get("audit_summary")
        if refs is None or summary is None:
            return

    if summary.needs_ocr_compendio:
        st.error("El PDF no contiene texto extraíble. needs_ocr_compendio=true.")
        st.stop()

    st.subheader("KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Presentes descargadas", len(summary.present_downloaded))
    col2.metric(
        "Faltan descargar pero aparecen en el compendio",
        len(summary.present_not_downloaded),
    )
    col3.metric(
        "No registradas (puede ser otro organismo)", len(summary.not_registered)
    )
    coverage_total = len(summary.present_downloaded) + len(summary.present_not_downloaded)
    coverage_pct = (
        (len(summary.present_downloaded) / coverage_total * 100)
        if coverage_total
        else 0.0
    )
    st.caption(
        f"Cobertura compendio: {len(summary.present_downloaded)} / {coverage_total} "
        f"({coverage_pct:.1f}%)."
    )

    st.subheader("Faltan descargar pero aparecen en el compendio")
    missing_rows = [
        {
            "doc_key": item.doc_key,
            "status": item.status,
            "last_checked_at": item.last_checked_at,
            "last_downloaded_at": item.last_downloaded_at,
            "url": item.url,
        }
        for item in summary.present_not_downloaded
    ]
    st.dataframe(maybe_dataframe(missing_rows))

    st.subheader("No registradas (refs)")
    not_registered_set = set(summary.not_registered)
    not_registered_rows = [
        {
            "page_number": ref.page_number,
            "confidence": ref.confidence,
            "raw_reference": ref.raw_reference,
            "evidence_snippet": ref.evidence_snippet,
        }
        for ref in refs
        if ref.doc_key_normalized in not_registered_set
    ]
    st.dataframe(maybe_dataframe(not_registered_rows))

    if st.session_state.get("audit_run_id") != summary.run_id:
        st.session_state["audit_run_id"] = summary.run_id
        st.session_state["missing_downloads_reviewed"] = []
        st.session_state["missing_downloads_atp"] = []

    st.subheader("Depurar missing_downloads con Ollama")
    enable_review = st.toggle("Activar LLM (Ollama local)", value=False)
    if enable_review:
        base_url = st.text_input(
            "Base URL Ollama",
            value="http://localhost:11434",
            help="Ej: http://localhost:11434 o http://localhost:11434/api/chat",
        )
        model = st.selectbox(
            "Modelo",
            [
                "llama3.1:70b-instruct-q4_K_M",
                "llava:13b",
                "qwen2.5:14b-instruct-q5_K_M",
                "qwen2.5:7b-instruct",
            ],
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.number_input(
            "Max tokens (num_predict)", min_value=0, value=256, step=32
        )
        confidence_threshold = st.slider(
            "Confidence threshold (ATP_MISSING)", 0.0, 1.0, 0.8, 0.05
        )
        run_review = st.button("Run Review")

        if run_review:
            config = OllamaConfig(
                model=model,
                temperature=float(temperature),
                max_tokens=int(max_tokens) if max_tokens else None,
                base_url=base_url,
            )
            reviewer = OllamaReviewer(OllamaClient(config))
            try:
                reviewed, atp_only = review_missing_downloads(
                    summary.present_not_downloaded,
                    refs,
                    reviewer,
                    Path(summary.export_dir),
                    summary.run_id,
                    model_name=model,
                    confidence_threshold=confidence_threshold,
                    db_path=db_path,
                    save_to_db=save_to_db,
                )
            except OllamaUnavailableError as exc:
                st.error(str(exc))
                reviewed = []
                atp_only = []
            else:
                summary = update_audit_summary_with_review(summary, reviewed, atp_only)
                st.success(
                    "Review completado. Exportados missing_downloads_reviewed.csv y "
                    "missing_downloads_atp.csv."
                )

            st.session_state["missing_downloads_reviewed"] = reviewed
            st.session_state["missing_downloads_atp"] = atp_only

        reviewed_cached = st.session_state.get("missing_downloads_reviewed", [])
        atp_cached = st.session_state.get("missing_downloads_atp", [])
        tabs = st.tabs(["ATP_MISSING", "OTHER_ORG", "DETECTION_ERROR", "UNKNOWN"])
        verdict_map = {
            "ATP_MISSING": tabs[0],
            "OTHER_ORG": tabs[1],
            "DETECTION_ERROR": tabs[2],
            "UNKNOWN": tabs[3],
        }
        for verdict, tab in verdict_map.items():
            with tab:
                rows = [
                    review.as_dict()
                    for review in reviewed_cached
                    if review.verdict == verdict
                ]
                st.dataframe(maybe_dataframe(rows))
                if rows:
                    st.download_button(
                        f"Export {verdict}",
                        data=rows_to_csv(rows),
                        file_name=f"missing_downloads_{verdict.lower()}.csv",
                    )
        if atp_cached:
            atp_rows = [review.as_dict() for review in atp_cached]
            st.download_button(
                "Export solo ATP_MISSING",
                data=rows_to_csv(atp_rows),
                file_name="missing_downloads_atp.csv",
            )

    st.caption(f"Exportado en {summary.export_dir} con run_id={summary.run_id}.")


def render_normas(db_path: Path) -> None:
    st.title("Normas (Leyes/Decretos/CA)")
    seed_tab, upload_tab, resolve_tab = st.tabs(
        ["Seed catálogo", "Upload PDF", "Resolver"]
    )

    with seed_tab:
        st.subheader("Seed catálogo")
        default_seed_path = data_dir() / "state" / "seeds" / "norms.yml"
        seed_path_value = st.text_input(
            "Ruta seeds YAML", value=str(default_seed_path)
        )
        if st.button("Cargar seeds"):
            try:
                summary = seed_catalog(
                    seed_path=Path(seed_path_value), db_path=db_path
                )
                st.success("Seeds cargados.")
                st.json(summary)
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

    with upload_tab:
        st.subheader("Upload PDF")
        with st.form("norm_upload_form"):
            norm_key = st.text_input("Norm key")
            norm_type = st.text_input("Norm type (opcional)")
            source_kind = st.selectbox(
                "Source kind",
                [
                    "CONSOLIDATED_CURRENT",
                    "CONSOLIDATED_HISTORIC",
                    "OFFICIAL_GAZETTE",
                    "OTHER",
                ],
            )
            authoritative = st.checkbox("Authoritative", value=False)
            notes = st.text_area("Notas", value="")
            file = st.file_uploader("PDF", type=["pdf"])
            submit = st.form_submit_button("Subir")

        if submit:
            if not norm_key:
                st.error("Debe ingresar norm_key.")
            elif not file:
                st.error("Debe cargar un PDF.")
            else:
                payload = upload_norm_pdf_ui(
                    db_path=db_path,
                    base_dir=data_dir(),
                    norm_key=norm_key,
                    file_bytes=file.getvalue(),
                    original_filename=file.name,
                    source_kind=source_kind,
                    authoritative=authoritative,
                    notes=notes or None,
                    norm_type=norm_type or None,
                )
                st.success("Upload completado.")
                st.json(payload)

    with resolve_tab:
        st.subheader("Resolver texto libre")
        query = st.text_input("Texto a resolver")
        if st.button("Resolver"):
            if not query.strip():
                st.error("Debe ingresar un texto.")
            else:
                result = resolve_norm_ui(db_path=db_path, text=query)
                if not result.get("match"):
                    st.warning("No resuelto.")
                st.json(result)



def render_config() -> None:
    st.title("Config")
    config = load_config(config_path())

    with st.form("config_form"):
        base_url_new = st.text_input("base_url_new", value=config.base_url_new)
        base_url_old = st.text_input("base_url_old", value=config.base_url_old)
        years_raw = st.text_input("years (coma separada)", value=", ".join(map(str, config.years)))
        max_n_raw = st.text_area(
            "max_n_by_year (JSON)",
            value=json.dumps(config.max_n_by_year, indent=2, ensure_ascii=False),
        )
        old_range_cols = st.columns(2)
        with old_range_cols[0]:
            old_start = st.number_input("old_range.start", value=config.old_range.start, min_value=1)
        with old_range_cols[1]:
            old_end = st.number_input("old_range.end", value=config.old_range.end, min_value=1)
        rate_limit = st.number_input(
            "rate_limit_rps",
            value=config.rate_limit_rps,
            min_value=1,
        )
        text_cols = st.columns(3)
        with text_cols[0]:
            min_chars_total = st.number_input(
                "text_quality.min_chars_total",
                value=config.text_quality.min_chars_total,
                min_value=0,
            )
        with text_cols[1]:
            min_chars_per_page = st.number_input(
                "text_quality.min_chars_per_page",
                value=config.text_quality.min_chars_per_page,
                min_value=0,
            )
        with text_cols[2]:
            min_alpha_ratio = st.number_input(
                "text_quality.min_alpha_ratio",
                value=float(config.text_quality.min_alpha_ratio),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        save = st.form_submit_button("Guardar")

    if save:
        try:
            updated = config.model_dump()
            updated.update(
                {
                    "base_url_new": base_url_new,
                    "base_url_old": base_url_old,
                    "years": parse_years(years_raw),
                    "max_n_by_year": parse_max_n_by_year(max_n_raw),
                    "old_range": {"start": int(old_start), "end": int(old_end)},
                    "rate_limit_rps": int(rate_limit),
                    "text_quality": {
                        "min_chars_total": int(min_chars_total),
                        "min_chars_per_page": int(min_chars_per_page),
                        "min_alpha_ratio": float(min_alpha_ratio),
                    },
                }
            )
            new_config = Config.model_validate(updated)
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Error al parsear config: {exc}")
        else:
            save_config(new_config, config_path())
            st.success("Config guardada.")

    if st.button("Validar"):
        try:
            validate_config_state(config_path(), state_path())
        except Exception as exc:  # noqa: BLE001 - show config validation errors.
            st.error(f"Error de validación: {exc}")
        else:
            st.success("Config OK")



def render_logs() -> None:
    st.title("Logs")
    logs_dir = data_dir() / "logs"
    if not logs_dir.exists():
        st.info("No hay logs aún.")
        return
    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        st.info("No hay logs aún.")
        return

    log_file = st.selectbox("Archivo", [path.name for path in log_files])
    line_count = st.number_input("Líneas", value=200, min_value=10, max_value=2000, step=10)

    if st.button("Refresh"):
        st.cache_data.clear()

    selected_path = logs_dir / log_file
    tail = cached_log_tail(str(selected_path), int(line_count))
    st.code(tail, language="text")


if __name__ == "__main__":
    run_app()
