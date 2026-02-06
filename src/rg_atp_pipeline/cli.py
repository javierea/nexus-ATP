"""CLI for rg_atp_pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from .config import Config, default_config, load_config, save_config
from .logging_utils import setup_logging
from .paths import config_path, data_dir, state_path
from .planner import plan_all
from .state import State, default_state, load_state, save_state

app = typer.Typer(help="rg_atp_pipeline CLI (Etapa 0)")



def ensure_dirs() -> None:
    """Ensure required directory structure exists."""
    base = data_dir()
    for sub in [
        "raw_pdfs",
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


@app.callback()
def main() -> None:
    """Global options for rg_atp_pipeline."""
