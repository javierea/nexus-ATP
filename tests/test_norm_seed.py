from pathlib import Path

from rg_atp_pipeline.services.norm_seed import seed_norms_from_yaml
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def test_seed_norms_from_yaml(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    seeds_path = tmp_path / "norms.yml"
    seeds_path.write_text(
        """
- norm_key: LEY-83-F
  norm_type: LEY
  jurisdiction: CHACO
  title: "Código Tributario Provincial"
  aliases:
    - alias_text: "Ley 83-F"
      alias_kind: NUMBERED
      confidence: 1.0
    - alias_text: "Dec. Ley 2444/62"
      alias_kind: LEGACY
      confidence: 1.0
""".strip()
    )

    summary = seed_norms_from_yaml(db_path=db_path, seeds_path=seeds_path)
    assert summary == {"norms": 1, "aliases": 2, "sources": 0}

    repo = NormsRepository(db_path)
    match = repo.resolve_norm_by_alias("Dec. Ley 2444/62")
    assert match is not None
    _, norm_key, confidence, alias_text = match
    assert norm_key == "LEY-83-F"
    assert confidence == 1.0
    assert alias_text == "Dec. Ley 2444/62"
