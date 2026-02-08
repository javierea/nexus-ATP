"""Project bootstrap helpers."""

from __future__ import annotations

from .config import default_config, save_config
from .paths import config_path, data_dir, state_path
from .state import default_state, save_state
from .storage_sqlite import DocumentStore


def ensure_dirs() -> None:
    """Ensure required directory structure exists."""
    base = data_dir()
    for sub in [
        "raw_pdfs",
        "raw_pdfs/latest",
        "text",
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
