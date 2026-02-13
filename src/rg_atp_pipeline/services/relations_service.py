"""Stage 4.1 relation typing from Stage 4 citation links."""

from __future__ import annotations

import logging
import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rg_atp_pipeline.config import load_config
from rg_atp_pipeline.paths import config_path
from rg_atp_pipeline.services.relation_extractor import RelationCandidate, extract_relation_candidates
from rg_atp_pipeline.services.relation_llm import verify_relation_candidates
from rg_atp_pipeline.storage.migrations import ensure_schema


@dataclass(frozen=True)
class StructuredUnit:
    unit_id: str
    unit_type: str | None
    text: str


def run_relations(
    db_path: Path,
    data_dir: Path,
    doc_keys: list[str] | None,
    limit_docs: int | None,
    llm_mode: str,
    min_confidence: float,
    prompt_version: str,
    batch_size: int,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    """Run relation typing from existing citations and links."""
    ensure_schema(db_path)
    logger = logging.getLogger("rg_atp_pipeline.relations")
    config = load_config(config_path())
    llm_mode = str(llm_mode).strip().lower()
    model = ollama_model or config.ollama_model
    base_url = ollama_base_url or config.ollama_base_url
    timeout_sec = config.llm_timeout_sec

    docs_processed = 0
    links_seen = 0
    relations_inserted = 0
    llm_verified = 0
    unknown_count = 0
    errors: list[str] = []
    total_candidates_detected = 0
    candidates_inserted_now = 0
    gated_count = 0
    skipped_already_reviewed_count = 0
    batches_sent = 0
    by_type_detected: dict[str, int] = {}
    by_type_inserted: dict[str, int] = {}

    logger.info(
        "Stage 4.1 LLM config: llm_mode=%r threshold/min_confidence=%s batch_size=%s prompt_version=%s model=%s",
        llm_mode,
        min_confidence,
        batch_size,
        prompt_version,
        model,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        rows = _load_citation_links(conn, doc_keys=doc_keys, limit_docs=limit_docs)
        docs_processed = len({row["source_doc_key"] for row in rows})
        pending_llm: list[dict[str, Any]] = []
        units_cache: dict[str, dict[str, StructuredUnit]] = {}

        for row in rows:
            links_seen += 1
            doc_key = str(row["source_doc_key"])
            if doc_key not in units_cache:
                units_cache[doc_key] = _load_structured_units(data_dir, doc_key)
            text = _build_local_text(row, units_cache[doc_key])
            candidates = extract_relation_candidates(text)
            total_candidates_detected += len(candidates)
            for candidate in candidates:
                by_type_detected[candidate.relation_type] = by_type_detected.get(candidate.relation_type, 0) + 1

                should_verify = llm_mode == "verify" and 0.6 <= candidate.confidence < min_confidence
                should_insert = candidate.confidence >= min_confidence or should_verify
                if not should_insert:
                    continue

                method = "REGEX"
                if should_verify:
                    method = "MIXED"
                relation_id, inserted = _insert_relation_extraction(
                    conn,
                    row,
                    candidate,
                    method=method,
                )
                if inserted:
                    relations_inserted += 1
                    candidates_inserted_now += 1
                    by_type_inserted[candidate.relation_type] = by_type_inserted.get(candidate.relation_type, 0) + 1

                final_type = candidate.relation_type
                final_conf = candidate.confidence
                if should_verify and relation_id is not None:
                    if _has_relation_review(
                        conn,
                        relation_id=relation_id,
                        llm_model=model,
                        prompt_version=prompt_version,
                    ):
                        skipped_already_reviewed_count += 1
                        continue
                    pending_llm.append(
                        {
                            "relation_id": relation_id,
                            "candidate_id": str(relation_id),
                            "citation_raw_text": row["raw_text"],
                            "evidence_snippet": candidate.evidence_snippet,
                            "unit_text": text,
                            "target_norm_key": row["target_norm_key"],
                            "regex_relation_type": candidate.relation_type,
                            "regex_scope": candidate.scope,
                            "regex_scope_detail": candidate.scope_detail,
                            "regex_confidence": candidate.confidence,
                        }
                    )
                if final_type == "UNKNOWN" or final_conf < 0.5:
                    unknown_count += 1

        gated_count = len(pending_llm)
        logger.info(
            "Stage 4.1 candidates: total_candidates_detected=%s candidates_inserted_now=%s gated_count=%s skipped_already_reviewed_count=%s",
            total_candidates_detected,
            candidates_inserted_now,
            gated_count,
            skipped_already_reviewed_count,
        )

        if llm_mode == "verify" and pending_llm:
            for start in range(0, len(pending_llm), batch_size):
                batch = pending_llm[start : start + batch_size]
                batches_sent += 1
                payload = [
                    {
                        "candidate_id": item["candidate_id"],
                        "citation_raw_text": item["citation_raw_text"],
                        "evidence_snippet": item["evidence_snippet"],
                        "unit_text": item["unit_text"],
                        "target_norm_key": item["target_norm_key"],
                    }
                    for item in batch
                ]
                try:
                    results = verify_relation_candidates(
                        payload,
                        model=model,
                        base_url=base_url,
                        prompt_version=prompt_version,
                        timeout_sec=timeout_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                    logger.error("Error LLM relations: %s", exc)
                    continue
                parsed = {_safe_int(item.get("candidate_id")): item for item in results}
                for item in batch:
                    relation_id = item["relation_id"]
                    result = parsed.get(relation_id)
                    if not result:
                        continue
                    normalized = _normalize_llm_result(result)
                    if not normalized:
                        continue
                    inserted = _insert_relation_review(
                        conn,
                        relation_id=relation_id,
                        llm_model=model,
                        prompt_version=prompt_version,
                        review=normalized,
                    )
                    if inserted:
                        llm_verified += 1
                    _update_relation_with_review(conn, relation_id, normalized)

        unknown_count = by_type_inserted.get("UNKNOWN", 0)
        logger.info(
            "Stage 4.1 LLM: batches_sent=%s llm_verified=%s",
            batches_sent,
            llm_verified,
        )
        conn.commit()

    return {
        "docs_processed": docs_processed,
        "links_seen": links_seen,
        "relations_inserted": relations_inserted,
        "candidates_inserted_now": candidates_inserted_now,
        "total_candidates_detected": total_candidates_detected,
        "llm_verified": llm_verified,
        "unknown_count": unknown_count,
        "by_type": by_type_inserted,
        "by_type_inserted": by_type_inserted,
        "by_type_detected": by_type_detected,
        "llm_mode_effective": llm_mode,
        "threshold_min_confidence": min_confidence,
        "batch_size": batch_size,
        "prompt_version": prompt_version,
        "model": model,
        "gated_count": gated_count,
        "skipped_already_reviewed_count": skipped_already_reviewed_count,
        "batches_sent": batches_sent,
        "errors": errors,
    }


def _load_citation_links(
    conn: sqlite3.Connection,
    doc_keys: list[str] | None,
    limit_docs: int | None,
) -> list[sqlite3.Row]:
    where = ["l.resolution_status IN ('RESOLVED', 'PLACEHOLDER_CREATED')"]
    params: list[Any] = []
    if doc_keys:
        where.append(f"c.source_doc_key IN ({','.join('?' for _ in doc_keys)})")
        params.extend(doc_keys)
    query = (
        "SELECT c.citation_id, c.source_doc_key, c.source_unit_id, c.raw_text, "
        "c.evidence_snippet, l.link_id, l.target_norm_key, l.resolution_status "
        "FROM citations c JOIN citation_links l ON l.citation_id = c.citation_id "
        f"WHERE {' AND '.join(where)} ORDER BY c.source_doc_key, c.citation_id"
    )
    rows = conn.execute(query, params).fetchall()
    if limit_docs is None:
        return rows
    accepted_docs: set[str] = set()
    limited: list[sqlite3.Row] = []
    for row in rows:
        key = row["source_doc_key"]
        if key not in accepted_docs and len(accepted_docs) >= limit_docs:
            continue
        accepted_docs.add(key)
        limited.append(row)
    return limited


def _build_local_text(
    row: sqlite3.Row,
    structured_units: dict[str, StructuredUnit],
) -> str:
    source_unit_id = str(row["source_unit_id"] or "").strip()
    if source_unit_id and source_unit_id in structured_units:
        text = structured_units[source_unit_id].text.strip()
        if text:
            return text
    parts = [row["raw_text"] or "", row["evidence_snippet"] or ""]
    return "\n".join(part for part in parts if part).strip()


def _load_structured_units(data_dir: Path, source_doc_key: str) -> dict[str, StructuredUnit]:
    structured_path = data_dir / "structured" / f"{source_doc_key}.json"
    if not structured_path.exists():
        return {}
    try:
        payload = json.loads(structured_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    result: dict[str, StructuredUnit] = {}
    for item in payload.get("units", []) or []:
        text = str(item.get("text") or "").strip()
        unit_id = str(item.get("unit_id") or item.get("id") or "").strip()
        if not unit_id or not text:
            continue
        result[unit_id] = StructuredUnit(
            unit_id=unit_id,
            unit_type=str(item.get("unit_type") or item.get("type") or "") or None,
            text=text,
        )
    return result


def _insert_relation_extraction(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    candidate: RelationCandidate,
    method: str,
) -> tuple[int | None, bool]:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO relation_extractions (
            citation_id, link_id, source_doc_key, target_norm_key,
            relation_type, direction, scope, scope_detail,
            method, confidence, evidence_snippet, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["citation_id"],
            row["link_id"],
            row["source_doc_key"],
            row["target_norm_key"],
            candidate.relation_type,
            candidate.direction,
            candidate.scope,
            candidate.scope_detail,
            method,
            candidate.confidence,
            candidate.evidence_snippet,
            candidate.explanation[:150],
            now,
        ),
    )
    if cur.rowcount:
        relation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return relation_id, True
    existing = conn.execute(
        """
        SELECT relation_id
        FROM relation_extractions
        WHERE citation_id = ? AND link_id = ? AND relation_type = ?
          AND scope = ? AND COALESCE(scope_detail, '') = COALESCE(?, '')
          AND method = ?
        """,
        (
            row["citation_id"],
            row["link_id"],
            candidate.relation_type,
            candidate.scope,
            candidate.scope_detail,
            method,
        ),
    ).fetchone()
    return (int(existing[0]), False) if existing else (None, False)


def _normalize_llm_result(item: dict[str, Any]) -> dict[str, Any] | None:
    relation_type = str(item.get("relation_type", "UNKNOWN")).upper()
    direction = str(item.get("direction", "UNKNOWN")).upper()
    scope = str(item.get("scope", "UNKNOWN")).upper()
    scope_detail = item.get("scope_detail")
    if isinstance(scope_detail, str):
        scope_detail = scope_detail.strip() or None
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    explanation = str(item.get("explanation", "Sin explicación.")).strip()[:120]
    return {
        "relation_type": relation_type,
        "direction": direction,
        "scope": scope,
        "scope_detail": scope_detail,
        "confidence": confidence,
        "explanation": explanation,
    }


def _insert_relation_review(
    conn: sqlite3.Connection,
    relation_id: int,
    llm_model: str,
    prompt_version: str,
    review: dict[str, Any],
) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO relation_llm_reviews (
            relation_id, llm_model, prompt_version,
            relation_type, direction, scope, scope_detail,
            llm_confidence, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id,
            llm_model,
            prompt_version,
            review["relation_type"],
            review["direction"],
            review["scope"],
            review["scope_detail"],
            review["confidence"],
            review["explanation"],
            now,
        ),
    )
    return bool(cur.rowcount)


def _has_relation_review(
    conn: sqlite3.Connection,
    relation_id: int,
    llm_model: str,
    prompt_version: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM relation_llm_reviews
        WHERE relation_id = ? AND llm_model = ? AND prompt_version = ?
        LIMIT 1
        """,
        (relation_id, llm_model, prompt_version),
    ).fetchone()
    return row is not None


def _update_relation_with_review(
    conn: sqlite3.Connection,
    relation_id: int,
    review: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE relation_extractions
        SET relation_type = ?, direction = ?, scope = ?, scope_detail = ?,
            confidence = ?, explanation = ?
        WHERE relation_id = ?
        """,
        (
            review["relation_type"],
            review["direction"],
            review["scope"],
            review["scope_detail"],
            review["confidence"],
            review["explanation"],
            relation_id,
        ),
    )


def _safe_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1
