"""Seed helpers for norms catalog."""

from __future__ import annotations

from pathlib import Path

import yaml

from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def seed_norms_from_yaml(db_path: Path, seeds_path: Path) -> dict[str, int]:
    """Seed norms and aliases from a YAML list."""
    ensure_schema(db_path)
    if not seeds_path.exists():
        raise FileNotFoundError(f"Archivo de seeds no encontrado: {seeds_path}")

    data = yaml.safe_load(seeds_path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError("El YAML de seeds debe ser una lista.")

    repo = NormsRepository(db_path)
    summary = {"norms": 0, "aliases": 0, "sources": 0}
    for entry in data:
        norm_id = repo.upsert_norm(
            norm_key=entry["norm_key"],
            norm_type=entry["norm_type"],
            jurisdiction=entry.get("jurisdiction"),
            year=entry.get("year"),
            number=entry.get("number"),
            suffix=entry.get("suffix"),
            title=entry.get("title"),
        )
        summary["norms"] += 1
        for alias in entry.get("aliases", []):
            repo.add_alias(
                norm_id=norm_id,
                alias_text=alias["alias_text"],
                alias_kind=alias.get("alias_kind", "OTHER"),
                confidence=float(alias.get("confidence", 1.0)),
                valid_from=alias.get("valid_from"),
                valid_to=alias.get("valid_to"),
            )
            summary["aliases"] += 1
        source_url = entry.get("source_url")
        if source_url:
            repo.get_or_create_source(
                norm_id=norm_id,
                source_kind=entry.get("source_kind", "OTHER"),
                source_method="url_fetch",
                url=source_url,
                is_authoritative=bool(entry.get("is_authoritative", False)),
                notes=entry.get("source_notes"),
            )
            repo.set_norm_status(norm_id, "HAS_SOURCE")
            summary["sources"] += 1

    return summary
