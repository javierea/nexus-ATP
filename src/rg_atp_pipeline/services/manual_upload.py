"""Manual upload service for norm PDFs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


@dataclass(frozen=True)
class UploadSummary:
    """Result summary for a manual upload."""

    norm_key: str
    sha256: str
    destination: str
    version_inserted: bool
    latest_pointer: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_norm_key(norm_key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in norm_key)


def _write_latest_pointer(pointer_path: Path, target: Path) -> str:
    if pointer_path.exists() or pointer_path.is_symlink():
        if pointer_path.is_dir():
            shutil.rmtree(pointer_path)
        else:
            pointer_path.unlink()
    try:
        os.symlink(target, pointer_path)
        return str(pointer_path)
    except OSError:
        pointer_path = pointer_path.with_suffix(".json")
        payload = {"latest_path": str(target)}
        pointer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return str(pointer_path)


def upload_norm_pdf(
    db_path: Path,
    base_dir: Path,
    norm_key: str,
    file_path: Path,
    source_kind: str,
    is_authoritative: bool,
    notes: str | None,
    norm_type: str | None = None,
) -> UploadSummary:
    """Upload a PDF for a norm and register versioned source."""
    ensure_schema(db_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    repo = NormsRepository(db_path)
    norm_id = repo.upsert_norm(
        norm_key=norm_key,
        norm_type=norm_type or "OTRO",
    )
    source_id = repo.get_or_create_source(
        norm_id=norm_id,
        source_kind=source_kind,
        source_method="manual_upload",
        url=None,
        is_authoritative=is_authoritative,
        notes=notes,
    )

    sha256 = _sha256_file(file_path)
    sanitized = _sanitize_norm_key(norm_key)
    raw_dir = base_dir / "raw_pdfs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{sanitized}__{sha256}.pdf"
    if not destination.exists():
        shutil.copy2(file_path, destination)

    version_inserted = repo.insert_source_version(
        source_id=source_id,
        sha256=sha256,
        downloaded_at=_utc_now(),
        pdf_path=str(destination),
        file_size_bytes=destination.stat().st_size,
    )
    repo.set_norm_status(norm_id, "DOWNLOADED")

    latest_dir = raw_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_pointer = _write_latest_pointer(
        latest_dir / f"{sanitized}.pdf",
        destination,
    )

    return UploadSummary(
        norm_key=norm_key,
        sha256=sha256,
        destination=str(destination),
        version_inserted=version_inserted,
        latest_pointer=latest_pointer,
    )
