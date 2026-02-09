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


def render_citations_stage(db_path: Path) -> None:
    st.title("Etapa 4 — Citas normativas")
    config = load_config(config_path())
    data_root = data_dir()

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
        llm_mode = st.radio(
            "Modo LLM",
            ["off", "verify"],
            index=0,
            horizontal=True,
        )

        ollama_base_url = config.ollama_base_url
        ollama_model = config.ollama_model
        batch_size = config.llm_batch_size

        if llm_mode == "verify":
            st.markdown("**Overrides Ollama (opcional)**")
            ollama_base_url = st.text_input(
                "Base URL Ollama",
                value=config.ollama_base_url,
            )
            ollama_model = st.text_input(
                "Modelo Ollama",
                value=config.ollama_model,
            )
            batch_size = st.number_input(
                "Batch size LLM",
                min_value=1,
                value=config.llm_batch_size,
                step=1,
            )
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
        try:
            with st.spinner("Ejecutando Etapa 4..."):
                summary = run_citations(
                    db_path=db_path,
                    data_dir=data_root,
                    doc_keys=doc_keys,
                    limit_docs=limit_docs_value,
                    llm_mode=llm_mode,
                    min_confidence=min_confidence,
                    create_placeholders=create_placeholders,
                    batch_size=int(batch_size) if llm_mode == "verify" else None,
                    ollama_model=ollama_model if llm_mode == "verify" else None,
                    ollama_base_url=ollama_base_url if llm_mode == "verify" else None,
                )
        except Exception as exc:  # noqa: BLE001 - show runtime errors.
            st.error(f"Error al ejecutar Etapa 4: {exc}")
        else:
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
) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    query = """
        SELECT
            citation_id,
            source_doc_key,
            source_unit_id,
            source_unit_type,
            raw_text,
            norm_type_guess,
            norm_key_candidate,
            regex_confidence,
            detected_at
        FROM citations
    """
    params: list[Any] = []
    if doc_keys:
        placeholders = ", ".join("?" for _ in doc_keys)
        query += f" WHERE source_doc_key IN ({placeholders})"
        params.extend(doc_keys)
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
