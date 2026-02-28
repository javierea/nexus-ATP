from pathlib import Path

from rg_atp_pipeline.rg_detector import detect_rg_starts, export_rg_splits, split_rg_boundaries


SAMPLE = """===PAGE 46===
RESOLUCIÓN GENERAL Nº 1234
VISTO:
Texto de la primera RG.
ARTÍCULO 1°.- ...
===PAGE 47===
Más texto.
===PAGE 48===
RESOLUCIÓN GENERAL N° 2345/24
VISTO Y CONSIDERANDO:
Texto de la segunda RG.
"""


def test_detect_rg_starts_accepts_old_and_slashed_formats() -> None:
    starts = detect_rg_starts(SAMPLE, logical_page_offset=45)
    assert len(starts) == 2
    assert starts[0].rg_number_raw == "1234"
    assert starts[0].visto_found is True
    assert starts[0].start_page_real == 46
    assert starts[0].start_page_logical == 1
    assert starts[1].rg_number_raw == "2345/24"


def test_split_rg_boundaries_uses_next_header_as_end() -> None:
    boundaries = split_rg_boundaries(SAMPLE, logical_page_offset=45)
    assert len(boundaries) == 2
    assert boundaries[0].start_page_real == 46
    assert boundaries[0].end_page_real == 47
    assert "Texto de la primera RG" in boundaries[0].text
    assert boundaries[1].start_page_logical == 3


def test_export_rg_splits_writes_files(tmp_path: Path) -> None:
    summary = export_rg_splits(SAMPLE, tmp_path, logical_page_offset=45, skip_existing=False)
    assert summary.exported == 2
    assert (tmp_path / "RG-1234.txt").exists()
    assert (tmp_path / "RG-2345-24.txt").exists()
