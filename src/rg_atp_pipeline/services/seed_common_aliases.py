"""Seed common aliases used for norm resolution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def _alias_exists(
    conn: sqlite3.Connection,
    norm_id: int,
    alias_text: str,
    alias_kind: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM norm_aliases
        WHERE norm_id = ?
          AND alias_text = ?
          AND alias_kind = ?
        """,
        (norm_id, alias_text, alias_kind),
    ).fetchone()
    return row is not None


def seed_common_aliases(db_path: Path, seed_path: Path) -> dict[str, int]:
    """Seed common aliases from YAML in an idempotent way."""
    ensure_schema(db_path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Archivo de seeds no encontrado: {seed_path}")

    data = yaml.safe_load(seed_path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError("El YAML de common aliases debe ser una lista.")

    repo = NormsRepository(db_path)
    summary = {
        "norms_upserted": 0,
        "aliases_inserted": 0,
        "aliases_skipped": 0,
        "errors": 0,
    }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for entry in data:
            try:
                norm_id = repo.upsert_norm(
                    norm_key=entry["norm_key"],
                    norm_type=entry["norm_type"],
                    jurisdiction=entry.get("jurisdiction"),
                    year=entry.get("year"),
                    number=entry.get("number"),
                    suffix=entry.get("suffix"),
                    title=entry.get("title"),
                )
                summary["norms_upserted"] += 1
            except (KeyError, ValueError, TypeError):
                summary["errors"] += 1
                continue

            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list):
                summary["errors"] += 1
                continue

            for alias in aliases:
                try:
                    alias_text = str(alias["alias_text"])
                    alias_kind = str(alias.get("alias_kind", "OTHER"))
                    confidence = float(alias.get("confidence", 1.0))
                    valid_from = alias.get("valid_from")
                    valid_to = alias.get("valid_to")
                except (KeyError, TypeError, ValueError):
                    summary["errors"] += 1
                    continue

                if _alias_exists(
                    conn=conn,
                    norm_id=norm_id,
                    alias_text=alias_text,
                    alias_kind=alias_kind,
                ):
                    summary["aliases_skipped"] += 1
                    continue

                try:
                    repo.add_alias(
                        norm_id=norm_id,
                        alias_text=alias_text,
                        alias_kind=alias_kind,
                        confidence=confidence,
                        valid_from=valid_from,
                        valid_to=valid_to,
                    )
                    summary["aliases_inserted"] += 1
                except sqlite3.DatabaseError:
                    summary["errors"] += 1

    return summary
