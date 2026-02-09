from pathlib import Path

from rg_atp_pipeline.norms_ui import resolve_norm_ui, seed_catalog, upload_norm_pdf_ui


def _pdf_bytes(text: str) -> bytes:
    content = (
        f"%PDF-1.4\n% {text}\n1 0 obj<</Type/Catalog>>endobj\n"
        "trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    return content.encode("utf-8")


def test_seed_catalog_and_resolve(tmp_path: Path) -> None:
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

    summary = seed_catalog(seed_path=seeds_path, db_path=db_path)
    assert summary["norms"] == 1

    result = resolve_norm_ui(db_path=db_path, text="Dec. Ley 2444/62")
    assert result["match"] is not None
    assert result["match"]["norm_key"] == "LEY-83-F"


def test_upload_norm_pdf_ui_versions(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    base_dir = tmp_path
    payload1 = upload_norm_pdf_ui(
        db_path=db_path,
        base_dir=base_dir,
        norm_key="LEY-83-F",
        file_bytes=_pdf_bytes("version 1"),
        original_filename="ley.pdf",
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v1",
        norm_type="LEY",
    )
    assert payload1["version_inserted"] is True
    assert Path(payload1["pdf_path"]).exists()

    payload1_repeat = upload_norm_pdf_ui(
        db_path=db_path,
        base_dir=base_dir,
        norm_key="LEY-83-F",
        file_bytes=_pdf_bytes("version 1"),
        original_filename="ley.pdf",
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v1",
        norm_type="LEY",
    )
    assert payload1_repeat["version_inserted"] is False

    payload2 = upload_norm_pdf_ui(
        db_path=db_path,
        base_dir=base_dir,
        norm_key="LEY-83-F",
        file_bytes=_pdf_bytes("version 2"),
        original_filename="ley.pdf",
        source_kind="CONSOLIDATED_CURRENT",
        authoritative=True,
        notes="v2",
        norm_type="LEY",
    )
    assert payload2["version_inserted"] is True
    assert payload2["sha256"] != payload1["sha256"]
