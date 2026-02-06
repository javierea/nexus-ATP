from pathlib import Path

from rg_atp_pipeline.cli import init_project, validate_config_state
from rg_atp_pipeline.config import default_config, save_config
from rg_atp_pipeline.paths import config_path, data_dir, state_path
from rg_atp_pipeline.state import default_state, save_state


def test_init_creates_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))

    init_project()

    assert config_path().exists()
    assert state_path().exists()
    assert (data_dir() / "logs").exists()


def test_validate_config_ok(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))

    save_config(default_config(), config_path())
    save_state(default_state(), state_path())

    config, state = validate_config_state(config_path(), state_path())
    assert config.user_agent == "rg_atp_pipeline/0.1"
    assert state.schema_version == 1
