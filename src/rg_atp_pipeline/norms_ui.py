"""UI-friendly helpers for norms catalog."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

from rg_atp_pipeline.services.manual_upload import upload_norm_pdf
from rg_atp_pipeline.services.norm_seed import seed_norms_from_yaml
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def seed_catalog(seed_path: Path, db_path: Path) -> dict[str, int]:
    """Seed norms catalog from a YAML file."""
    return seed_norms_from_yaml(db_path=db_path, seeds_path=seed_path)


def upload_norm_pdf_ui(
    db_path: Path,
    base_dir: Path,
    norm_key: str,
    file_bytes: bytes,
    original_filename: str,
    source_kind: str,
    authoritative: bool,
    notes: str | None,
    norm_type: str | None = None,
) -> dict[str, object]:
    """Upload a norm PDF from UI bytes and return a summary dict."""
    tmp_dir = base_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(
        dir=tmp_dir,
        suffix=suffix,
        prefix="norm_upload_",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(file_bytes)

    try:
        summary = upload_norm_pdf(
            db_path=db_path,
            base_dir=base_dir,
            norm_key=norm_key,
            file_path=tmp_path,
            source_kind=source_kind,
            is_authoritative=authoritative,
            notes=notes,
            norm_type=norm_type,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    payload = asdict(summary)
    payload["pdf_path"] = payload.pop("destination")
    return payload


def resolve_norm_ui(db_path: Path, text: str) -> dict[str, object]:
    """Resolve a norm from free text using aliases."""
    repo = NormsRepository(db_path)
    match = repo.resolve_norm_by_alias(text)
    if not match:
        return {"query": text, "match": None}
    norm_id, norm_key, confidence, alias_text = match
    return {
        "query": text,
        "match": {
            "norm_id": norm_id,
            "norm_key": norm_key,
            "confidence": confidence,
            "matched_alias": alias_text,
        },
    }
