"""CLI for rg_atp_pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from pydantic import ValidationError

from .config import Config, default_config, load_config, save_config
from .fetcher import FetchOptions, run_fetch
from .logging_utils import setup_logging
from .paths import config_path, data_dir, state_path
from .planner import plan_all
from .storage_sqlite import DocumentStore
from .state import State, default_state, load_state, save_state

app = typer.Typer(help="rg_atp_pipeline CLI (Etapa 0)")



def ensure_dirs() -> None:
    """Ensure required directory structure exists."""
    base = data_dir()
    for sub in [
        "raw_pdfs",
        "raw_pdfs/latest",
        "raw_text",
        "structured",
        "state",
        "logs",
        "tmp",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)



def init_project() -> None:
    """Initialize folders, config, and state if missing."""
    ensure_dirs()

    cfg_path = config_path()
    if not cfg_path.exists():
        save_config(default_config(), cfg_path)

    st_path = state_path()
    if not st_path.exists():
        save_state(default_state(), st_path)

    store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
    store.initialize()



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


@app.callback()
def main() -> None:
    """Global options for rg_atp_pipeline."""
