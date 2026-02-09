"""CLI for rg_atp_pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from .config import Config, load_config
from .audit_compendio import run_audit_compendio
from .fetcher import FetchOptions, run_fetch
from .logging_utils import setup_logging
from .paths import config_path, data_dir, state_path
from .planner import plan_all
from .project import init_project
from .services.manual_upload import upload_norm_pdf
from .storage.migrations import ensure_schema
from .storage.norms_repo import NormsRepository
from .storage_sqlite import DocumentStore
from .state import State, load_state
from .structure_segmenter import StructureOptions, run_structure
from .text_extractor import ExtractOptions, run_extract
from .web_ui import run_ui

app = typer.Typer(help="rg_atp_pipeline CLI (Etapa 3)")



def validate_config_state(cfg_path: Path, st_path: Path) -> tuple[Config, State]:
    """Validate config and state files, raising on error."""
    config = load_config(cfg_path)
    state = load_state(st_path)
    return config, state


@app.command()
def init() -> None:
    """Create folders and default config/state if missing."""
    setup_logging(data_dir() / "logs")
    init_project()
    typer.echo("Inicialización completada.")


@app.command("validate")
def validate_cmd() -> None:
    """Validate config.yml and data/state/state.json."""
    try:
        validate_config_state(config_path(), state_path())
    except (ValidationError, FileNotFoundError, json.JSONDecodeError) as exc:
        typer.secho(f"Error de validación: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo("OK")


@app.command("show-config")
def show_config() -> None:
    """Print parsed config."""
    config = load_config(config_path())
    typer.echo(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))


@app.command("show-state")
def show_state() -> None:
    """Print parsed state."""
    state = load_state(state_path())
    typer.echo(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))


@app.command("plan")
def plan() -> None:
    """Show planned URLs (no requests)."""
    config = load_config(config_path())
    urls = plan_all(config)
    typer.echo(f"Total URLs: {len(urls)}")
    if urls:
        head = urls[:10]
        tail = urls[-10:] if len(urls) > 10 else []
        typer.echo("Primeras 10:")
        for url in head:
            typer.echo(f"- {url}")
        if tail:
            typer.echo("Últimas 10:")
            for url in tail:
                typer.echo(f"- {url}")


@app.command("fetch")
def fetch(
    mode: str = typer.Option("both", help="Modo de descarga: new, old o both."),
    year: int | None = typer.Option(None, help="Filtrar por año (solo modo new)."),
    n_start: int | None = typer.Option(None, help="Inicio de N para modo new."),
    n_end: int | None = typer.Option(None, help="Fin de N para modo new."),
    old_start: int | None = typer.Option(None, help="Inicio para modo old."),
    old_end: int | None = typer.Option(None, help="Fin para modo old."),
    dry_run: bool = typer.Option(False, help="No realiza requests, solo planifica."),
    max_downloads: int | None = typer.Option(None, "--max", help="Máximo de descargas."),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing/--no-skip-existing",
        help="Omitir entradas ya descargadas con archivo disponible.",
    ),
) -> None:
    """Fetch PDFs (Etapa 1)."""
    setup_logging(data_dir() / "logs")
    init_project()
    config = load_config(config_path())
    state = load_state(state_path())
    store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
    summary = run_fetch(
        config,
        state,
        store,
        data_dir(),
        FetchOptions(
            mode=mode,
            year=year,
            n_start=n_start,
            n_end=n_end,
            old_start=old_start,
            old_end=old_end,
            dry_run=dry_run,
            max_downloads=max_downloads,
            skip_existing=skip_existing,
        ),
        logging.getLogger("rg_atp_pipeline"),
    )
    typer.echo(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))


@app.command("extract")
def extract(
    doc_key: str | None = typer.Option(None, help="Procesar un doc_key específico."),
    status: str = typer.Option("DOWNLOADED", help="Status de documentos a procesar."),
    limit: int | None = typer.Option(None, help="Máximo de documentos a procesar."),
    force: bool = typer.Option(False, help="Reprocesar aunque ya exista texto."),
    only_text: bool = typer.Option(False, help="Solo documentos con text_status=EXTRACTED."),
    only_needs_ocr: bool = typer.Option(False, help="Solo documentos con text_status=NEEDS_OCR."),
) -> None:
    """Extract raw text from PDFs (Etapa 2)."""
    setup_logging(data_dir() / "logs")
    init_project()
    config = load_config(config_path())
    store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
    summary = run_extract(
        config,
        store,
        data_dir(),
        ExtractOptions(
            status=status,
            limit=limit,
            doc_key=doc_key,
            force=force,
            only_text=only_text,
            only_needs_ocr=only_needs_ocr,
        ),
        logging.getLogger("rg_atp_pipeline"),
    )
    typer.echo(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))


@app.command("audit-compendio")
def audit_compendio(
    pdf_path: str = typer.Option(
        str(data_dir() / "compendio-legislativo-al-31-12-2024.pdf"),
        "--pdf-path",
        help="Ruta al PDF del compendio legislativo.",
    ),
    export_dir: str = typer.Option(
        str(data_dir() / "audit"),
        "--export-dir",
        help="Directorio de exportación (CSV/JSON).",
    ),
    min_confidence: float = typer.Option(
        0.0,
        "--min-confidence",
        help="Confianza mínima para incluir referencias.",
    ),
    save_to_db: bool = typer.Option(
        True,
        "--save-to-db/--no-save-to-db",
        help="Guardar referencias en tabla compendio_refs.",
    ),
    only_missing_downloads: bool = typer.Option(
        False,
        "--only-missing-downloads",
        help="Exportar solo missing_downloads a CSV.",
    ),
) -> None:
    """Auditar el compendio legislativo para detectar RG faltantes."""
    setup_logging(data_dir() / "logs")
    init_project()
    db_path = data_dir() / "state" / "rg_atp.sqlite"
    refs, summary = run_audit_compendio(
        Path(pdf_path),
        db_path,
        Path(export_dir),
        min_confidence=min_confidence,
        save_to_db=save_to_db,
        export_refs=not only_missing_downloads,
        export_missing_downloads=True,
    )

    if summary.needs_ocr_compendio:
        typer.secho(
            "El PDF no contiene texto extraíble. "
            "needs_ocr_compendio=true. Abortando auditoría.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"Detectadas {summary.total_refs_detected} refs "
        f"({summary.unique_refs_detected} únicas)."
    )
    typer.echo(
        f"Presentes descargadas: {len(summary.present_downloaded)} | "
        f"Presentes no descargadas: {len(summary.present_not_downloaded)} | "
        f"No registradas: {len(summary.not_registered)}"
    )
    typer.echo(
        json.dumps(
            summary.as_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", help="Host para el servidor UI."),
    port: int = typer.Option(8000, help="Puerto para el servidor UI."),
) -> None:
    """Launch minimal web UI for inventory and actions."""
    setup_logging(data_dir() / "logs")
    init_project()
    run_ui(host, port)


@app.command("ui-streamlit")
def ui_streamlit(
    host: str = typer.Option("127.0.0.1", help="Host para la UI Streamlit."),
    port: int = typer.Option(8501, help="Puerto para la UI Streamlit."),
) -> None:
    """Launch Streamlit control panel."""
    setup_logging(data_dir() / "logs")
    init_project()
    app_path = Path(__file__).with_name("streamlit_app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    subprocess.run(cmd, check=False)


@app.command("structure")
def structure(
    doc_key: str | None = typer.Option(None, help="Procesar un doc_key específico."),
    limit: int | None = typer.Option(None, help="Máximo de documentos a procesar."),
    force: bool = typer.Option(False, help="Reprocesar aunque ya exista estructura."),
    include_needs_ocr: bool = typer.Option(
        False,
        "--include-needs-ocr",
        help="Incluir documentos NEEDS_OCR (por defecto solo EXTRACTED).",
    ),
    export_json: bool = typer.Option(
        True,
        "--export-json/--no-export-json",
        help="Exportar JSON estructurado por documento.",
    ),
) -> None:
    """Segmentar texto crudo en unidades (Etapa 3)."""
    setup_logging(data_dir() / "logs")
    init_project()
    store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
    summary = run_structure(
        store,
        data_dir(),
        StructureOptions(
            doc_key=doc_key,
            limit=limit,
            force=force,
            include_needs_ocr=include_needs_ocr,
            export_json=export_json,
        ),
        logging.getLogger("rg_atp_pipeline"),
    )
    typer.echo(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))


@app.command("seed-norms")
def seed_norms() -> None:
    """Seed norms and aliases from data/state/seeds/norms.yml."""
    setup_logging(data_dir() / "logs")
    init_project()
    db_path = data_dir() / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    seeds_path = data_dir() / "state" / "seeds" / "norms.yml"
    if not seeds_path.exists():
        typer.secho(
            f"Archivo de seeds no encontrado: {seeds_path}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    data = yaml.safe_load(seeds_path.read_text()) or []
    if not isinstance(data, list):
        typer.secho("El YAML de seeds debe ser una lista.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    repo = NormsRepository(db_path)
    summary = {"norms": 0, "aliases": 0, "sources": 0}
    for entry in data:
        norm_id = repo.upsert_norm(
            norm_key=entry["norm_key"],
            norm_type=entry["norm_type"],
            jurisdiction=entry.get("jurisdiction"),
            year=entry.get("year"),
            number=entry.get("number"),
            suffix=entry.get("suffix"),
            title=entry.get("title"),
        )
        summary["norms"] += 1
        for alias in entry.get("aliases", []):
            repo.add_alias(
                norm_id=norm_id,
                alias_text=alias["alias_text"],
                alias_kind=alias.get("alias_kind", "OTHER"),
                confidence=float(alias.get("confidence", 1.0)),
                valid_from=alias.get("valid_from"),
                valid_to=alias.get("valid_to"),
            )
            summary["aliases"] += 1
        source_url = entry.get("source_url")
        if source_url:
            repo.get_or_create_source(
                norm_id=norm_id,
                source_kind=entry.get("source_kind", "OTHER"),
                source_method="url_fetch",
                url=source_url,
                is_authoritative=bool(entry.get("is_authoritative", False)),
                notes=entry.get("source_notes"),
            )
            repo.set_norm_status(norm_id, "HAS_SOURCE")
            summary["sources"] += 1
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command("upload-norm")
def upload_norm(
    norm_key: str = typer.Option(..., "--norm-key", help="Clave canónica de norma."),
    file: Path = typer.Option(..., "--file", help="Ruta al PDF."),
    source_kind: str = typer.Option(
        "CONSOLIDATED_CURRENT",
        "--source-kind",
        help="Tipo de fuente.",
    ),
    authoritative: bool = typer.Option(
        False,
        "--authoritative/--no-authoritative",
        help="Marca la fuente como prioritaria.",
    ),
    notes: str | None = typer.Option(None, "--notes", help="Notas opcionales."),
    norm_type: str | None = typer.Option(
        None,
        "--norm-type",
        help="Tipo de norma si hay que crear placeholder.",
    ),
) -> None:
    """Upload manual PDF for a norm and register versioned source."""
    setup_logging(data_dir() / "logs")
    init_project()
    summary = upload_norm_pdf(
        db_path=data_dir() / "state" / "rg_atp.sqlite",
        base_dir=data_dir(),
        norm_key=norm_key,
        file_path=file,
        source_kind=source_kind,
        is_authoritative=authoritative,
        notes=notes,
        norm_type=norm_type,
    )
    typer.echo(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))


@app.command("resolve-norm")
def resolve_norm(
    text: str = typer.Argument("", help="Texto libre a resolver."),
    text_option: str | None = typer.Option(None, "--text", help="Texto libre."),
) -> None:
    """Resolve a norm based on alias text."""
    setup_logging(data_dir() / "logs")
    init_project()
    query = text_option or text
    if not query:
        typer.secho("Debe proporcionar texto a resolver.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    repo = NormsRepository(data_dir() / "state" / "rg_atp.sqlite")
    match = repo.resolve_norm_by_alias(query)
    if not match:
        typer.echo(
            json.dumps(
                {
                    "query": query,
                    "match": None,
                    "message": "No se encontró coincidencia. "
                    "Sugerido crear placeholder.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise typer.Exit(code=1)
    norm_id, norm_key, confidence, alias_text = match
    typer.echo(
        json.dumps(
            {
                "query": query,
                "norm_id": norm_id,
                "norm_key": norm_key,
                "confidence": confidence,
                "matched_alias": alias_text,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.callback()
def main() -> None:
    """Global options for rg_atp_pipeline."""
