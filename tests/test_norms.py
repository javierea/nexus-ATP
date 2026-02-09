import hashlib
import json
import os
import sqlite3
from pathlib import Path

from reportlab.pdfgen import canvas

from rg_atp_pipeline.cli import seed_norms, upload_norm
from rg_atp_pipeline.paths import data_dir
from rg_atp_pipeline.project import init_project
from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def _create_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, text)
    c.save()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ensure_schema_creates_tables_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "norms" in tables
        assert "norm_aliases" in tables
        assert "norm_sources" in tables
        assert "norm_source_versions" in tables
        norms_indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(norms)")
        }
        alias_indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(norm_aliases)")
        }
        assert "idx_norms_norm_key" in norms_indexes
        assert "idx_norm_aliases_alias_text" in alias_indexes


def test_seed_norms_and_resolve(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    init_project()
    seeds_dir = data_dir() / "state" / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = seeds_dir / "norms.yml"
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

    seed_norms()

    repo = NormsRepository(data_dir() / "state" / "rg_atp.sqlite")
    match = repo.resolve_norm_by_alias("Dec. Ley 2444/62")
    assert match is not None
    _, norm_key, confidence, alias_text = match
    assert norm_key == "LEY-83-F"
    assert confidence == 1.0
    assert alias_text == "Dec. Ley 2444/62"


def test_upload_norm_versions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    init_project()
    pdf1 = tmp_path / "one.pdf"
    pdf2 = tmp_path / "two.pdf"
    _create_pdf(pdf1, "version 1")
    _create_pdf(pdf2, "version 2")

    upload_norm(
        norm_key="LEY-83-F",
        file=pdf1,
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v1",
        norm_type="LEY",
    )

    db_path = data_dir() / "state" / "rg_atp.sqlite"
    sha1 = _sha256(pdf1)
    expected_path = data_dir() / "raw_pdfs" / f"LEY-83-F__{sha1}.pdf"
    assert expected_path.exists()
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM norm_source_versions"
        ).fetchone()[0]
        assert count == 1

    upload_norm(
        norm_key="LEY-83-F",
        file=pdf1,
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v1",
        norm_type="LEY",
    )
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM norm_source_versions"
        ).fetchone()[0]
        assert count == 1

    upload_norm(
        norm_key="LEY-83-F",
        file=pdf2,
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v2",
        norm_type="LEY",
    )
    sha2 = _sha256(pdf2)
    expected_path2 = data_dir() / "raw_pdfs" / f"LEY-83-F__{sha2}.pdf"
    assert expected_path2.exists()
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM norm_source_versions"
        ).fetchone()[0]
        assert count == 2

    latest_dir = data_dir() / "raw_pdfs" / "latest"
    pointer_pdf = latest_dir / "LEY-83-F.pdf"
    pointer_json = latest_dir / "LEY-83-F.json"
    if pointer_pdf.exists() or pointer_pdf.is_symlink():
        target = Path(os.readlink(pointer_pdf)) if pointer_pdf.is_symlink() else pointer_pdf
        assert target.name == expected_path2.name
    else:
        payload = json.loads(pointer_json.read_text())
        assert Path(payload["latest_path"]).name == expected_path2.name
