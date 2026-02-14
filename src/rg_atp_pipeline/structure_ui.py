"""UI-facing helpers for Etapa 3 (structure)."""

from __future__ import annotations

from pathlib import Path

from rg_atp_pipeline.queries import (
    get_structure_anomalies,
    get_structure_summary,
    get_units_for_doc,
)
from rg_atp_pipeline.structure_segmenter import StructureOptions, run_structure
from rg_atp_pipeline.storage_sqlite import DocumentStore

__all__ = [
    "run_structure_ui",
    "get_structure_summary",
    "get_structure_anomalies",
    "get_units_for_doc",
]


def run_structure_ui(
    store: DocumentStore,
    data_dir: Path,
    doc_key: str | None,
    limit: int | None,
    force: bool,
    include_needs_ocr: bool,
    export_json: bool,
    logger,
) -> dict[str, int]:
    """Execute Stage 3 structure segmentation and return serializable summary."""
    summary = run_structure(
        store,
        data_dir,
        StructureOptions(
            doc_key=doc_key,
            limit=limit,
            force=force,
            include_needs_ocr=include_needs_ocr,
            export_json=export_json,
        ),
        logger,
    )
    return summary.as_dict()


