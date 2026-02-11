from pathlib import Path

from rg_atp_pipeline.norms_ui import resolve_norm_ui
from rg_atp_pipeline.services.seed_common_aliases import seed_common_aliases
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def test_seed_common_aliases_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    seed_path = Path("data/state/seeds/common_aliases.yml")

    first = seed_common_aliases(db_path=db_path, seed_path=seed_path)
    assert first["norms_upserted"] == 2
    assert first["aliases_inserted"] == 22
    assert first["aliases_skipped"] == 0
    assert first["errors"] == 0

    second = seed_common_aliases(db_path=db_path, seed_path=seed_path)
    assert second["norms_upserted"] == 2
    assert second["aliases_inserted"] == 0
    assert second["aliases_skipped"] == 22
    assert second["errors"] == 0


def test_resolve_common_aliases_with_punctuation_and_historical_text(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    seed_path = Path("data/state/seeds/common_aliases.yml")
    seed_common_aliases(db_path=db_path, seed_path=seed_path)

    repo = NormsRepository(db_path)
    by_abbrev = repo.resolve_norm_by_alias("Cod. Trib. Provincial")
    assert by_abbrev is not None
    _, norm_key, _, _ = by_abbrev
    assert norm_key == "LEY-83-F"

    ui_match = resolve_norm_ui(db_path=db_path, text="Ley de facto 2071")
    assert ui_match["match"] is not None
    assert ui_match["match"]["norm_key"] == "LEY-TARIFARIA-CHACO"
