"""Stage 4 citation extraction and resolution service."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from rg_atp_pipeline.config import load_config
from rg_atp_pipeline.paths import config_path
from rg_atp_pipeline.services.llm_verifier import verify_candidates
from rg_atp_pipeline.services.reference_extractor import Candidate, extract_candidates
from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository
from rg_atp_pipeline.utils.normalize import normalize_text


@dataclass(frozen=True)
class UnitPayload:
    unit_id: str | None
    unit_type: str | None
    text: str


@dataclass(frozen=True)
class CitationPayload:
    citation_id: int
    candidate: Candidate
    source_doc_key: str
    source_unit_id: str | None
    source_unit_type: str | None


def run_citations(
    db_path: Path,
    data_dir: Path,
    doc_keys: list[str] | None,
    limit_docs: int | None,
    llm_mode: str,
    min_confidence: float,
    create_placeholders: bool,
    batch_size: int | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    prompt_version: str | None = None,
    llm_gate_regex_threshold: float | None = None,
    llm_timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run citation extraction and normalization."""
    ensure_schema(db_path)
    logger = logging.getLogger("rg_atp_pipeline.citations")
    config = load_config(config_path())

    batch_size = batch_size or config.llm_batch_size
    ollama_model = ollama_model or config.ollama_model
    ollama_base_url = ollama_base_url or config.ollama_base_url
    prompt_version = prompt_version or config.llm_prompt_version
    llm_gate_regex_threshold = (
        llm_gate_regex_threshold or config.llm_gate_regex_threshold
    )
    llm_timeout_sec = llm_timeout_sec or config.llm_timeout_sec

    available_docs = _collect_doc_keys(data_dir, doc_keys)
    if limit_docs is not None:
        available_docs = available_docs[:limit_docs]

    candidates_total = 0
    citations_inserted = 0
    llm_verified = 0
    rejected = 0
    resolved = 0
    placeholders_created = 0
    unresolved = 0
    errors = 0

    repo = NormsRepository(db_path)
    citations: list[CitationPayload] = []

    with _connect(db_path) as conn:
        for doc_key in available_docs:
            units = _load_units(data_dir, doc_key, logger)
            if not units:
                logger.warning("Sin texto disponible para %s.", doc_key)
                errors += 1
                continue
            for unit in units:
                if not unit.text.strip():
                    continue
                extracted = extract_candidates(unit.text)
                candidates_total += len(extracted)
                for candidate in extracted:
                    citation_id, inserted = _get_or_create_citation_id(
                        conn,
                        doc_key,
                        unit,
                        candidate,
                    )
                    if inserted:
                        citations_inserted += 1
                    citations.append(
                        CitationPayload(
                            citation_id=citation_id,
                            candidate=candidate,
                            source_doc_key=doc_key,
                            source_unit_id=unit.unit_id,
                            source_unit_type=unit.unit_type,
                        )
                    )

        reviews = {}
        if llm_mode == "verify":
            gated = [
                citation
                for citation in citations
                if _needs_llm_review(
                    citation.candidate,
                    llm_gate_regex_threshold,
                )
                and not _has_review(
                    conn,
                    citation.citation_id,
                    ollama_model,
                    prompt_version,
                )
            ]
            for batch in _chunked(gated, batch_size):
                payload = [
                    _candidate_payload(item)
                    for item in batch
                ]
                try:
                    results = verify_candidates(
                        payload,
                        model=ollama_model,
                        base_url=ollama_base_url,
                        prompt_version=prompt_version,
                        timeout_sec=llm_timeout_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error LLM: %s", exc)
                    errors += 1
                    continue
                for result in results:
                    review = _normalize_review(result)
                    if not review:
                        continue
                    citation_id = review["citation_id"]
                    try:
                        inserted = _insert_review(
                            conn,
                            citation_id,
                            ollama_model,
                            prompt_version,
                            review,
                        )
                    except sqlite3.IntegrityError as exc:
                        _log_integrity_error(
                            logger,
                            conn,
                            "citation_llm_reviews",
                            exc,
                            citation_id=citation_id,
                        )
                        raise
                    if inserted:
                        llm_verified += 1
                    reviews[citation_id] = review

        for citation in citations:
            review = reviews.get(citation.citation_id)
            decision = _decide_reference(
                citation,
                review,
                llm_mode,
                min_confidence,
            )
            if decision["status"] == "REJECTED":
                try:
                    _insert_link(
                        conn,
                        citation.citation_id,
                        None,
                        None,
                        "REJECTED",
                        decision["confidence"],
                    )
                except sqlite3.IntegrityError as exc:
                    _log_integrity_error(
                        logger,
                        conn,
                        "citation_links",
                        exc,
                        citation=citation,
                    )
                    raise
                rejected += 1
                continue

            norm_type = decision["norm_type"]
            key = decision["norm_key"]
            target_norm_id, target_norm_key = _resolve_norm(
                repo,
                key,
                citation.candidate.raw_text,
            )
            if target_norm_id is not None:
                try:
                    _insert_link(
                        conn,
                        citation.citation_id,
                        target_norm_id,
                        target_norm_key,
                        "RESOLVED",
                        decision["confidence"],
                    )
                except sqlite3.IntegrityError as exc:
                    _log_integrity_error(
                        logger,
                        conn,
                        "citation_links",
                        exc,
                        citation=citation,
                        target_norm_key=target_norm_key,
                        target_norm_id=target_norm_id,
                    )
                    raise
                resolved += 1
                continue

            if create_placeholders:
                placeholder_key = key or _make_placeholder_key(
                    citation.candidate.raw_text
                )
                target_norm_id = _get_or_create_norm_id_by_key(
                    conn,
                    placeholder_key,
                    norm_type or "OTRO",
                    status="PLACEHOLDER",
                )
                _add_norm_alias(
                    conn,
                    target_norm_id,
                    citation.candidate.raw_text,
                    alias_kind="CITATION",
                    confidence=0.4,
                )
                try:
                    _insert_link(
                        conn,
                        citation.citation_id,
                        target_norm_id,
                        placeholder_key,
                        "PLACEHOLDER_CREATED",
                        decision["confidence"],
                    )
                except sqlite3.IntegrityError as exc:
                    _log_integrity_error(
                        logger,
                        conn,
                        "citation_links",
                        exc,
                        citation=citation,
                        target_norm_key=placeholder_key,
                        target_norm_id=target_norm_id,
                    )
                    raise
                placeholders_created += 1
            else:
                try:
                    _insert_link(
                        conn,
                        citation.citation_id,
                        None,
                        None,
                        "UNRESOLVED",
                        decision["confidence"],
                    )
                except sqlite3.IntegrityError as exc:
                    _log_integrity_error(
                        logger,
                        conn,
                        "citation_links",
                        exc,
                        citation=citation,
                    )
                    raise
                unresolved += 1

    return {
        "docs_processed": len(available_docs),
        "candidates_total": candidates_total,
        "citations_inserted": citations_inserted,
        "llm_verified": llm_verified,
        "rejected": rejected,
        "resolved": resolved,
        "placeholders_created": placeholders_created,
        "unresolved": unresolved,
        "errors": errors,
    }


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _collect_doc_keys(data_dir: Path, doc_keys: list[str] | None) -> list[str]:
    if doc_keys:
        return doc_keys
    structured_dir = data_dir / "structured"
    raw_text_dir = data_dir / "raw_text"
    text_dir = data_dir / "text"
    keys: set[str] = set()
    if structured_dir.exists():
        for path in structured_dir.glob("*.json"):
            keys.add(path.stem)
    if raw_text_dir.exists():
        for path in raw_text_dir.glob("*.txt"):
            keys.add(path.stem)
    if not keys and text_dir.exists():
        for path in text_dir.glob("*.txt"):
            keys.add(path.stem)
    return sorted(keys)


def _load_units(data_dir: Path, doc_key: str, logger: logging.Logger) -> list[UnitPayload]:
    structured_path = data_dir / "structured" / f"{doc_key}.json"
    if structured_path.exists():
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("JSON estructurado inválido %s: %s", doc_key, exc)
            return []
        units = _units_from_structured(payload)
        if units:
            return units
    raw_path = data_dir / "raw_text" / f"{doc_key}.txt"
    if not raw_path.exists():
        raw_path = data_dir / "text" / f"{doc_key}.txt"
    if raw_path.exists():
        return [
            UnitPayload(
                unit_id=None,
                unit_type=None,
                text=raw_path.read_text(encoding="utf-8"),
            )
        ]
    return []


def _units_from_structured(payload: dict[str, Any]) -> list[UnitPayload]:
    units: list[UnitPayload] = []
    for item in payload.get("units", []) or []:
        text = item.get("text") or ""
        if text:
            units.append(
                UnitPayload(
                    unit_id=str(item.get("unit_id") or item.get("id") or ""),
                    unit_type=str(item.get("unit_type") or item.get("type") or ""),
                    text=text,
                )
            )
    if units:
        return units
    for group, unit_type in [("articles", "ARTICLE"), ("annexes", "ANNEX")]:
        for item in payload.get(group, []) or []:
            text = item.get("text") or item.get("body") or ""
            if text:
                unit_id = item.get("article_number") or item.get("annex_number") or ""
                units.append(
                    UnitPayload(
                        unit_id=str(unit_id),
                        unit_type=unit_type,
                        text=text,
                    )
                )
    if units:
        return units
    sections = payload.get("sections")
    if isinstance(sections, dict):
        combined = "\n".join(
            value for value in sections.values() if isinstance(value, str)
        ).strip()
        if combined:
            units.append(UnitPayload(unit_id=None, unit_type="SECTION", text=combined))
    return units


def _get_or_create_citation_id(
    conn: sqlite3.Connection,
    doc_key: str,
    unit: UnitPayload,
    candidate: Candidate,
) -> tuple[int, bool]:
    now = _utc_now()
    unit_id = unit.unit_id or ""
    params = (
        doc_key,
        unit_id,
        unit.unit_type or None,
        candidate.raw_text,
        candidate.norm_type_guess,
        candidate.norm_key_candidate,
        candidate.evidence_snippet,
        candidate.regex_confidence,
        now,
    )
    inserted = False
    try:
        row = conn.execute(
            """
            INSERT INTO citations (
                source_doc_key,
                source_unit_id,
                source_unit_type,
                raw_text,
                norm_type_guess,
                norm_key_candidate,
                evidence_snippet,
                regex_confidence,
                detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_doc_key,
                source_unit_id,
                raw_text,
                evidence_snippet
            ) DO NOTHING
            RETURNING citation_id
            """,
            params,
        ).fetchone()
        if row:
            return int(row["citation_id"]), True
    except sqlite3.OperationalError as exc:
        if "RETURNING" not in str(exc).upper():
            raise
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO citations (
                source_doc_key,
                source_unit_id,
                source_unit_type,
                raw_text,
                norm_type_guess,
                norm_key_candidate,
                evidence_snippet,
                regex_confidence,
                detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        inserted = cursor.rowcount == 1
    row = conn.execute(
        """
        SELECT citation_id
        FROM citations
        WHERE source_doc_key = ?
          AND COALESCE(source_unit_id, '') = COALESCE(?, '')
          AND raw_text = ?
          AND evidence_snippet = ?
        """,
        (
            doc_key,
            unit_id,
            candidate.raw_text,
            candidate.evidence_snippet,
        ),
    ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError("No se pudo obtener citation_id.")
    return int(row["citation_id"]), inserted


def _needs_llm_review(candidate: Candidate, threshold: float) -> bool:
    if candidate.regex_confidence < threshold:
        return True
    if candidate.norm_key_candidate is None:
        return True
    if candidate.norm_type_guess in {"DECRETO", "RG_CA"}:
        return True
    return False


def _candidate_payload(citation: CitationPayload) -> dict[str, Any]:
    return {
        "candidate_id": str(citation.citation_id),
        "raw_text": citation.candidate.raw_text,
        "evidence_snippet": citation.candidate.evidence_snippet,
        "norm_type_guess": citation.candidate.norm_type_guess,
        "norm_key_candidate": citation.candidate.norm_key_candidate,
    }


def _normalize_review(result: dict[str, Any]) -> dict[str, Any] | None:
    candidate_id = result.get("candidate_id")
    if candidate_id is None:
        return None
    try:
        citation_id = int(candidate_id)
    except (TypeError, ValueError):
        return None
    normalized_key = result.get("normalized_key")
    return {
        "citation_id": citation_id,
        "is_reference": bool(result.get("is_reference")),
        "norm_type": str(result.get("norm_type") or "OTRO").upper(),
        "normalized_key": normalized_key if normalized_key else None,
        "confidence": float(result.get("confidence") or 0.0),
        "explanation": str(result.get("explanation") or "").strip()[:200],
    }


def _insert_review(
    conn: sqlite3.Connection,
    citation_id: int,
    model: str,
    prompt_version: str,
    review: dict[str, Any],
) -> bool:
    now = _utc_now()
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO citation_llm_reviews (
            citation_id,
            llm_model,
            prompt_version,
            is_reference,
            norm_type,
            normalized_key,
            llm_confidence,
            explanation,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            citation_id,
            model,
            prompt_version,
            int(review["is_reference"]),
            review["norm_type"],
            review["normalized_key"],
            review["confidence"],
            review["explanation"],
            now,
        ),
    )
    return cursor.rowcount == 1


def _has_review(
    conn: sqlite3.Connection,
    citation_id: int,
    model: str,
    prompt_version: str,
) -> bool:
    row = conn.execute(
        """
        SELECT review_id
        FROM citation_llm_reviews
        WHERE citation_id = ? AND llm_model = ? AND prompt_version = ?
        """,
        (citation_id, model, prompt_version),
    ).fetchone()
    return row is not None


def _decide_reference(
    citation: CitationPayload,
    review: dict[str, Any] | None,
    llm_mode: str,
    min_confidence: float,
) -> dict[str, Any]:
    if llm_mode == "verify" and review is not None:
        if not review["is_reference"]:
            return {"status": "REJECTED", "confidence": review["confidence"]}
        key = review["normalized_key"] or citation.candidate.norm_key_candidate
        return {
            "status": "ACCEPTED",
            "norm_type": review["norm_type"],
            "norm_key": key,
            "confidence": review["confidence"],
        }
    if citation.candidate.regex_confidence < min_confidence:
        return {"status": "REJECTED", "confidence": citation.candidate.regex_confidence}
    return {
        "status": "ACCEPTED",
        "norm_type": citation.candidate.norm_type_guess,
        "norm_key": citation.candidate.norm_key_candidate,
        "confidence": citation.candidate.regex_confidence,
    }


def _resolve_norm(
    repo: NormsRepository,
    key: str | None,
    raw_text: str,
) -> tuple[int | None, str | None]:
    if key:
        norm = repo.get_norm(key)
        if norm:
            return norm.norm_id, norm.norm_key
    alias = repo.resolve_norm_by_alias(raw_text)
    if alias:
        norm_id, norm_key, _, _ = alias
        return norm_id, norm_key
    return None, None


def _make_placeholder_key(raw_text: str) -> str:
    normalized = normalize_text(raw_text)
    digest = sha1(normalized.encode("utf-8")).hexdigest()[:8].upper()
    return f"UNK-{digest}"


def _insert_link(
    conn: sqlite3.Connection,
    citation_id: int,
    target_norm_id: int | None,
    target_norm_key: str | None,
    status: str,
    confidence: float,
) -> None:
    if _has_link(conn, citation_id):
        return
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO citation_links (
            citation_id,
            target_norm_id,
            target_norm_key,
            resolution_status,
            resolution_confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            citation_id,
            target_norm_id,
            target_norm_key,
            status,
            confidence,
            now,
        ),
    )


def _has_link(conn: sqlite3.Connection, citation_id: int) -> bool:
    row = conn.execute(
        "SELECT link_id FROM citation_links WHERE citation_id = ?",
        (citation_id,),
    ).fetchone()
    return row is not None


def _chunked(items: list[CitationPayload], size: int) -> list[list[CitationPayload]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _add_norm_alias(
    conn: sqlite3.Connection,
    norm_id: int,
    alias_text: str,
    alias_kind: str = "OTHER",
    confidence: float = 1.0,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> None:
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO norm_aliases (
            norm_id,
            alias_text,
            alias_kind,
            confidence,
            valid_from,
            valid_to,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(norm_id, alias_text) DO UPDATE SET
            alias_kind = excluded.alias_kind,
            confidence = excluded.confidence,
            valid_from = excluded.valid_from,
            valid_to = excluded.valid_to
        """,
        (
            norm_id,
            alias_text,
            alias_kind,
            confidence,
            valid_from,
            valid_to,
            now,
        ),
    )


def _get_or_create_norm_id_by_key(
    conn: sqlite3.Connection,
    norm_key: str,
    norm_type: str,
    status: str,
) -> int:
    now = _utc_now()
    row = conn.execute(
        "SELECT norm_id FROM norms WHERE norm_key = ?",
        (norm_key,),
    ).fetchone()
    if row:
        return int(row["norm_id"])
    conn.execute(
        """
        INSERT INTO norms (
            norm_key,
            norm_type,
            jurisdiction,
            year,
            number,
            suffix,
            title,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)
        ON CONFLICT(norm_key) DO NOTHING
        """,
        (norm_key, norm_type, status, now, now),
    )
    row = conn.execute(
        "SELECT norm_id FROM norms WHERE norm_key = ?",
        (norm_key,),
    ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError("No se pudo obtener norm_id.")
    return int(row["norm_id"])


def _log_integrity_error(
    logger: logging.Logger,
    conn: sqlite3.Connection,
    table: str,
    exc: sqlite3.IntegrityError,
    citation: CitationPayload | None = None,
    citation_id: int | None = None,
    target_norm_key: str | None = None,
    target_norm_id: int | None = None,
) -> None:
    raw_text = citation.candidate.raw_text if citation else ""
    norm_key_candidate = citation.candidate.norm_key_candidate if citation else None
    source_doc_key = citation.source_doc_key if citation else None
    source_unit_id = citation.source_unit_id if citation else None
    citation_id = citation_id if citation_id is not None else (
        citation.citation_id if citation else None
    )
    logger.error(
        "IntegrityError en %s: %s | doc_key=%s unit_id=%s citation_id=%s "
        "norm_key_candidate=%s target_norm_key=%s target_norm_id=%s raw_text=%s",
        table,
        exc,
        source_doc_key,
        source_unit_id,
        citation_id,
        norm_key_candidate,
        target_norm_key,
        target_norm_id,
        raw_text[:160],
    )
    if logger.isEnabledFor(logging.DEBUG):
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        if rows:
            details = [
                f"{row['table']}({row['rowid']})->{row['parent']}({row['fkid']})"
                for row in rows
            ]
            logger.debug("foreign_key_check: %s", "; ".join(details))
        else:
            logger.debug("foreign_key_check: sin violaciones reportadas.")
