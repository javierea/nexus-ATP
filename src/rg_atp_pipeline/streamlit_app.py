"""Streamlit control panel for rg_atp_pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
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
from rg_atp_pipeline.ollama_client import (
    OllamaClient,
    OllamaConfig,
    OllamaReviewer,
    OllamaUnavailableError,
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
        ["Dashboard", "Fetch", "Extract", "Structure", "Audit", "Config", "Logs"],
    )

    if page == "Dashboard":
        render_dashboard(db_path)
    elif page == "Fetch":
        render_fetch(db_path, store, logger)
    elif page == "Extract":
        render_extract(db_path, store, logger)
    elif page == "Structure":
        render_structure(db_path, store, logger)
    elif page == "Audit":
        render_audit(db_path)
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

    if not submitted:
        return

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
