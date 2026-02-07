"""Fetch and store PDFs for rg_atp_pipeline (Etapa 1)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from hashlib import sha1
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import Config
from .http_client import HttpClient, HttpClientConfig, HttpClientError
from .planner import PlannedDoc, plan_new_docs, plan_old_docs
from .state import State, save_state
from .storage_sqlite import DocumentRecord, DocumentStore


@dataclass(frozen=True)
class FetchSummary:
    checked: int
    downloaded: int
    missing: int
    error: int
    changed_hash: int

    def as_dict(self) -> dict[str, int]:
        return {
            "checked": self.checked,
            "downloaded": self.downloaded,
            "missing": self.missing,
            "error": self.error,
            "changed_hash": self.changed_hash,
        }


@dataclass(frozen=True)
class FetchOptions:
    mode: str
    year: int | None
    n_start: int | None
    n_end: int | None
    old_start: int | None
    old_end: int | None
    dry_run: bool
    max_downloads: int | None
    skip_existing: bool


def run_fetch(
    config: Config,
    state: State,
    store: DocumentStore,
    data_dir: Path,
    options: FetchOptions,
    logger: logging.Logger,
) -> FetchSummary:
    store.initialize()
    planned = _plan_documents(config, options)
    logger.info("Planificadas %s URLs (modo=%s)", len(planned), options.mode)

    found_old_numbers: set[int] = set()
    if options.skip_existing:
        planned = _filter_existing(planned, store, found_old_numbers, logger)
        logger.info("Restantes %s URLs tras omitir descargadas.", len(planned))

    if options.dry_run:
        return FetchSummary(checked=len(planned), downloaded=0, missing=0, error=0, changed_hash=0)

    client = HttpClient(
        HttpClientConfig(
            rate_limit_rps=config.rate_limit_rps,
            timeout_sec=config.request_timeout_sec,
            max_attempts=config.retry.max_attempts,
            backoff_sec=config.retry.backoff_sec,
            user_agent=config.user_agent,
        ),
        logger=logger,
    )

    raw_dir = data_dir / "raw_pdfs"
    latest_dir = raw_dir / "latest"
    raw_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    checked = 0
    downloaded = 0
    missing = 0
    error = 0
    changed_hash = 0
    stop_due_to_max = False
    per_year_status: dict[int, list[str]] = {}
    old_number_cutoff: int | None = None
    current_old_number: int | None = None
    current_old_found = False

    for entry in planned:
        if entry.doc_family == "OLD":
            if old_number_cutoff is not None and entry.number > old_number_cutoff:
                continue
            if current_old_number is None or entry.number != current_old_number:
                if current_old_number is not None and not current_old_found:
                    missing += 1
                current_old_number = entry.number
                current_old_found = entry.number in found_old_numbers
            if entry.number in found_old_numbers:
                continue
        if options.max_downloads is not None and downloaded >= options.max_downloads:
            stop_due_to_max = True
            logger.info("Se alcanzó el máximo de descargas (%s).", options.max_downloads)
            break
        checked += 1
        result_status = "ERROR"
        http_status = None
        error_message = None
        latest_sha = store.get_latest_sha(entry.doc_key)
        now = _now_iso()

        try:
            status_code, _ = client.head(entry.url)
            http_status = status_code
            if status_code == 404:
                result_status = "MISSING"
                if entry.doc_family != "OLD":
                    missing += 1
                logger.info("MISSING %s", entry.url)
            elif status_code in (200, 405, 501):
                pdf_bytes = client.get_bytes(entry.url)
                sha256 = _hash_bytes(pdf_bytes)
                pdf_path = raw_dir / f"{entry.doc_key}__{sha256}.pdf"
                if not pdf_path.exists():
                    pdf_path.write_bytes(pdf_bytes)
                _write_latest_pointer(latest_dir, entry.doc_key, pdf_path)
                downloaded += 1
                result_status = "DOWNLOADED"
                if entry.doc_family == "OLD":
                    found_old_numbers.add(entry.number)
                    current_old_found = True
                    if old_number_cutoff is None or entry.number < old_number_cutoff:
                        old_number_cutoff = entry.number
                if latest_sha and latest_sha != sha256:
                    changed_hash += 1
                logger.info("DOWNLOADED %s -> %s", entry.doc_key, pdf_path.name)
                _record_document(
                    store,
                    entry,
                    now,
                    now,
                    sha256,
                    str(pdf_path),
                    result_status,
                    http_status,
                    error_message,
                )
                per_year_status.setdefault(entry.year or 0, []).append(result_status)
                continue
            else:
                result_status = "ERROR"
                error += 1
                error_message = f"HEAD {entry.url} -> {status_code}"
                logger.warning(error_message)
        except HttpClientError as exc:
            if exc.status_code == 404:
                result_status = "MISSING"
                http_status = 404
                if entry.doc_family != "OLD":
                    missing += 1
                logger.info("MISSING %s", entry.url)
            else:
                result_status = "ERROR"
                error += 1
                error_message = str(exc)
                logger.error("ERROR %s: %s", entry.url, exc)
        except OSError as exc:
            result_status = "ERROR"
            error += 1
            error_message = f"Filesystem error: {exc}"
            logger.error("ERROR %s: %s", entry.url, exc)

        _record_document(
            store,
            entry,
            now,
            None,
            None,
            None,
            result_status,
            http_status,
            error_message,
        )
        per_year_status.setdefault(entry.year or 0, []).append(result_status)

    if current_old_number is not None and not current_old_found:
        missing += 1

    summary = FetchSummary(
        checked=checked,
        downloaded=downloaded,
        missing=missing,
        error=error,
        changed_hash=changed_hash,
    )
    _update_state(state, data_dir, summary, planned, per_year_status, stop_due_to_max)
    return summary


def run_manual_fetch(
    config: Config,
    state: State,
    store: DocumentStore,
    data_dir: Path,
    url: str,
    logger: logging.Logger,
) -> FetchSummary:
    """Fetch a single PDF URL and register it as MANUAL."""
    store.initialize()
    client = HttpClient(
        HttpClientConfig(
            rate_limit_rps=config.rate_limit_rps,
            timeout_sec=config.request_timeout_sec,
            max_attempts=config.retry.max_attempts,
            backoff_sec=config.retry.backoff_sec,
            user_agent=config.user_agent,
        ),
        logger=logger,
    )

    raw_dir = data_dir / "raw_pdfs"
    latest_dir = raw_dir / "latest"
    raw_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    checked = 1
    downloaded = 0
    missing = 0
    error = 0
    changed_hash = 0

    now = _now_iso()
    doc_key = f"MANUAL-{sha1(url.encode('utf-8')).hexdigest()[:12]}"
    latest_sha = store.get_latest_sha(doc_key)
    result_status = "ERROR"
    http_status = None
    error_message = None

    try:
        status_code, _ = client.head(url)
        http_status = status_code
        if status_code == 404:
            result_status = "MISSING"
            missing += 1
            logger.info("MISSING %s", url)
        elif status_code in (200, 405, 501):
            pdf_bytes = client.get_bytes(url)
            sha256 = _hash_bytes(pdf_bytes)
            pdf_path = raw_dir / f"{doc_key}__{sha256}.pdf"
            if not pdf_path.exists():
                pdf_path.write_bytes(pdf_bytes)
            _write_latest_pointer(latest_dir, doc_key, pdf_path)
            downloaded += 1
            result_status = "DOWNLOADED"
            if latest_sha and latest_sha != sha256:
                changed_hash += 1
            logger.info("DOWNLOADED %s -> %s", doc_key, pdf_path.name)
            _record_manual_document(
                store,
                doc_key,
                url,
                now,
                now,
                sha256,
                str(pdf_path),
                result_status,
                http_status,
                error_message,
            )
        else:
            result_status = "ERROR"
            error += 1
            error_message = f"HEAD {url} -> {status_code}"
            logger.warning(error_message)
    except HttpClientError as exc:
        if exc.status_code == 404:
            result_status = "MISSING"
            http_status = 404
            missing += 1
            logger.info("MISSING %s", url)
        else:
            result_status = "ERROR"
            error += 1
            error_message = str(exc)
            logger.error("ERROR %s: %s", url, exc)
    except OSError as exc:
        result_status = "ERROR"
        error += 1
        error_message = f"Filesystem error: {exc}"
        logger.error("ERROR %s: %s", url, exc)

    if result_status != "DOWNLOADED":
        _record_manual_document(
            store,
            doc_key,
            url,
            now,
            None,
            None,
            None,
            result_status,
            http_status,
            error_message,
        )

    summary = FetchSummary(
        checked=checked,
        downloaded=downloaded,
        missing=missing,
        error=error,
        changed_hash=changed_hash,
    )
    _update_state(state, data_dir, summary, [], {}, True)
    return summary


def _plan_documents(config: Config, options: FetchOptions) -> list[PlannedDoc]:
    mode = options.mode.lower()
    docs: list[PlannedDoc] = []
    if mode in ("new", "both"):
        years = [options.year] if options.year is not None else config.years
        for year in years:
            max_n = config.max_n_by_year.get(str(year), 0)
            start_n = options.n_start if options.n_start is not None else 1
            end_n = options.n_end if options.n_end is not None else max_n
            docs.extend(plan_new_docs(config.base_url_new, year, start_n, end_n))
    if mode in ("old", "both"):
        start = options.old_start if options.old_start is not None else config.old_range.start
        end = options.old_end if options.old_end is not None else config.old_range.end
        docs.extend(
            plan_old_docs(
                config.base_url_old,
                start,
                end,
                config.old_min_number,
                config.old_year_start,
                config.old_year_end,
            )
        )
    return docs


def _filter_existing(
    planned: Iterable[PlannedDoc],
    store: DocumentStore,
    found_old_numbers: set[int],
    logger: logging.Logger,
) -> list[PlannedDoc]:
    filtered: list[PlannedDoc] = []
    skipped = 0
    for entry in planned:
        record = store.get_record(entry.doc_key)
        if (
            record
            and record.status == "DOWNLOADED"
            and record.latest_pdf_path
            and Path(record.latest_pdf_path).exists()
        ):
            skipped += 1
            if entry.doc_family == "OLD" and entry.number is not None:
                found_old_numbers.add(entry.number)
            continue
        filtered.append(entry)
    if skipped:
        logger.info("Omitidas %s entradas ya descargadas.", skipped)
    return filtered


def _record_document(
    store: DocumentStore,
    entry: PlannedDoc,
    last_checked_at: str,
    last_downloaded_at: str | None,
    latest_sha256: str | None,
    latest_pdf_path: str | None,
    status: str,
    http_status: int | None,
    error_message: str | None,
) -> None:
    record = DocumentRecord(
        doc_key=entry.doc_key,
        url=entry.url,
        doc_family=entry.doc_family,
        year=entry.year,
        number=entry.number,
        first_seen_at=last_checked_at,
        last_checked_at=last_checked_at,
        last_downloaded_at=last_downloaded_at,
        latest_sha256=latest_sha256,
        latest_pdf_path=latest_pdf_path,
        status=status,
        http_status=http_status,
        error_message=error_message,
    )
    store.upsert(record)


def _record_manual_document(
    store: DocumentStore,
    doc_key: str,
    url: str,
    last_checked_at: str,
    last_downloaded_at: str | None,
    latest_sha256: str | None,
    latest_pdf_path: str | None,
    status: str,
    http_status: int | None,
    error_message: str | None,
) -> None:
    record = DocumentRecord(
        doc_key=doc_key,
        url=url,
        doc_family="MANUAL",
        year=None,
        number=None,
        first_seen_at=last_checked_at,
        last_checked_at=last_checked_at,
        last_downloaded_at=last_downloaded_at,
        latest_sha256=latest_sha256,
        latest_pdf_path=latest_pdf_path,
        status=status,
        http_status=http_status,
        error_message=error_message,
    )
    store.upsert(record)


def _write_latest_pointer(latest_dir: Path, doc_key: str, target_path: Path) -> None:
    link_path = latest_dir / f"{doc_key}.pdf"
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(target_path, link_path)
    except (OSError, NotImplementedError):
        pointer_path = latest_dir / f"{doc_key}.json"
        payload = {"latest_pdf_path": str(target_path), "updated_at": _now_iso()}
        pointer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_state(
    state: State,
    data_dir: Path,
    summary: FetchSummary,
    planned: Iterable[PlannedDoc],
    per_year_status: dict[int, list[str]],
    stop_due_to_max: bool,
) -> None:
    state.last_run_at = datetime.now(timezone.utc)
    state.last_fetch_summary = summary.as_dict()

    if not stop_due_to_max:
        planned_by_year: dict[int, list[PlannedDoc]] = {}
        for entry in planned:
            if entry.doc_family == "NEW" and entry.year is not None:
                planned_by_year.setdefault(entry.year, []).append(entry)
        for year, entries in planned_by_year.items():
            statuses = per_year_status.get(year, [])
            if entries and len(statuses) == len(entries) and all(status == "DOWNLOADED" for status in statuses):
                last_n = max(entry.number for entry in entries)
                state.last_seen_n_by_year[str(year)] = last_n

    save_state(state, data_dir / "state" / "state.json")
