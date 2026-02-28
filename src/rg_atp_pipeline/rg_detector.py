"""Detect and split multiple RGs from a compendio text extract.

This stage is designed to run after `extract` and before `structure`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PAGE_MARKER_RE = re.compile(r"^===PAGE\s+(\d+)===$")
RG_HEADER_RE = re.compile(
    r"(?im)^\s*RESOLUCI[OÓ]N\s+GENERAL\s*(?:N[º°o]?\.?|N[UÚ]MERO\s*)?\s*[:\-]?\s*(\d{3,4}(?:/\d{2,4})?)\s*$"
)
VISTO_RE = re.compile(r"(?im)^\s*VISTO(?::|\s+Y\s+CONSIDERANDO:?|\s*$)")


@dataclass(frozen=True)
class RGStart:
    rg_number_raw: str
    start_line: int
    start_page_real: int
    start_page_logical: int | None
    header_line: str
    visto_found: bool


@dataclass(frozen=True)
class RGBoundary:
    rg_number_raw: str
    start_line: int
    end_line: int
    start_page_real: int
    end_page_real: int
    start_page_logical: int | None
    end_page_logical: int | None
    text: str


@dataclass(frozen=True)
class SplitSummary:
    starts_detected: int
    exported: int
    skipped_existing: int


def detect_rg_starts(raw_text: str, logical_page_offset: int = 0) -> list[RGStart]:
    """Detect potential RG starts using header + VISTO validation.

    logical_page_offset allows mapping when index numbering starts at a later PDF page.
    Example: if index page 1 == PDF page 46, use logical_page_offset=45.
    """

    lines = raw_text.splitlines()
    starts: list[RGStart] = []
    current_page = 1

    for idx, line in enumerate(lines):
        marker = PAGE_MARKER_RE.match(line.strip())
        if marker:
            current_page = int(marker.group(1))
            continue

        header = RG_HEADER_RE.match(line)
        if not header:
            continue

        window = "\n".join(lines[idx + 1 : idx + 6])
        visto_found = bool(VISTO_RE.search(window))
        logical_page = current_page - logical_page_offset if logical_page_offset else None
        starts.append(
            RGStart(
                rg_number_raw=header.group(1),
                start_line=idx + 1,
                start_page_real=current_page,
                start_page_logical=logical_page,
                header_line=line.strip(),
                visto_found=visto_found,
            )
        )

    return starts


def split_rg_boundaries(raw_text: str, logical_page_offset: int = 0) -> list[RGBoundary]:
    """Split raw text into RG blocks by detected starts.

    Keeps only starts that pass the VISTO/VISTO Y CONSIDERANDO check.
    """

    lines = raw_text.splitlines()
    starts = [item for item in detect_rg_starts(raw_text, logical_page_offset) if item.visto_found]
    if not starts:
        return []

    page_by_line = _page_lookup(lines)
    boundaries: list[RGBoundary] = []

    for i, start in enumerate(starts):
        start_idx = start.start_line - 1
        end_idx = (starts[i + 1].start_line - 2) if i + 1 < len(starts) else (len(lines) - 1)
        end_idx_effective = end_idx
        while end_idx_effective >= start_idx and PAGE_MARKER_RE.match(lines[end_idx_effective].strip()):
            end_idx_effective -= 1
        if end_idx_effective < start_idx:
            end_idx_effective = end_idx

        block = "\n".join(lines[start_idx : end_idx_effective + 1]).strip()
        end_page_real = page_by_line[end_idx_effective]
        end_page_logical = end_page_real - logical_page_offset if logical_page_offset else None
        boundaries.append(
            RGBoundary(
                rg_number_raw=start.rg_number_raw,
                start_line=start.start_line,
                end_line=end_idx_effective + 1,
                start_page_real=start.start_page_real,
                end_page_real=end_page_real,
                start_page_logical=start.start_page_logical,
                end_page_logical=end_page_logical,
                text=block,
            )
        )

    return boundaries


def export_rg_splits(
    raw_text: str,
    output_dir: Path,
    logical_page_offset: int = 0,
    skip_existing: bool = True,
) -> SplitSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = split_rg_boundaries(raw_text, logical_page_offset=logical_page_offset)
    exported = 0
    skipped_existing = 0

    for item in boundaries:
        filename = f"RG-{_slug(item.rg_number_raw)}.txt"
        out_path = output_dir / filename
        if skip_existing and out_path.exists():
            skipped_existing += 1
            continue
        out_path.write_text(item.text, encoding="utf-8")
        exported += 1

    return SplitSummary(
        starts_detected=len(boundaries),
        exported=exported,
        skipped_existing=skipped_existing,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", value).strip("-").upper()


def _page_lookup(lines: list[str]) -> list[int]:
    current_page = 1
    values: list[int] = []
    for line in lines:
        marker = PAGE_MARKER_RE.match(line.strip())
        if marker:
            current_page = int(marker.group(1))
        values.append(current_page)
    return values
