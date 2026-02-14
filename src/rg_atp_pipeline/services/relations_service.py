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
    only_structured: bool = False,
    extract_version: str = "relext-v2",
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
    skipped_according_to_no_target_now = 0
    inserted_according_to_with_target_now = 0
    collisions_merged_now = 0
    skipped_according_to_no_target_samples: list[dict[str, Any]] = []

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

        if only_structured and not doc_keys:
            doc_keys = [
                str(r["doc_key"])
                for r in conn.execute(
                    """
                    SELECT d.doc_key
                    FROM documents d
                    JOIN doc_structure s ON s.doc_key = d.doc_key
                    WHERE COALESCE(d.text_status, 'NONE') = 'EXTRACTED'
                      AND s.structure_status = 'STRUCTURED'
                    ORDER BY d.doc_key
                    """
                ).fetchall()
            ]
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

                if not should_verify:
                    allowed = _should_persist_relation(
                        relation_type=candidate.relation_type,
                        target_norm_key=row["target_norm_key"],
                    )
                    if not allowed:
                        skipped_according_to_no_target_now += 1
                        _append_skipped_sample(skipped_according_to_no_target_samples, row, candidate)
                        continue

                method = "REGEX"
                if should_verify:
                    method = "MIXED"
                relation_id, inserted = _insert_relation_extraction(
                    conn,
                    row,
                    candidate,
                    method=method,
                    extract_version=extract_version,
                )
                if inserted:
                    relations_inserted += 1
                    candidates_inserted_now += 1
                    by_type_inserted[candidate.relation_type] = by_type_inserted.get(candidate.relation_type, 0) + 1
                    if (
                        candidate.relation_type == "ACCORDING_TO"
                        and _has_target_norm_key(row["target_norm_key"])
                    ):
                        inserted_according_to_with_target_now += 1

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
                            "inserted_now": inserted,
                            "candidate_id": str(relation_id),
                            "row": row,
                            "candidate": candidate,
                            "citation_raw_text": row["raw_text"],
                            "evidence_snippet": candidate.evidence_snippet,
                            "unit_text": text,
                            "source_unit_number": row["source_unit_number"],
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

                    if not _should_persist_relation(
                        relation_type=normalized["relation_type"],
                        target_norm_key=item["target_norm_key"],
                    ):
                        if item.get("inserted_now"):
                            deleted = _delete_relation_extraction(conn, relation_id)
                            if deleted:
                                skipped_according_to_no_target_now += 1
                                _append_skipped_sample(
                                    skipped_according_to_no_target_samples,
                                    item["row"],
                                    item["candidate"],
                                )
                                relations_inserted = max(0, relations_inserted - 1)
                                candidates_inserted_now = max(0, candidates_inserted_now - 1)
                                prev = item["candidate"].relation_type
                                if by_type_inserted.get(prev):
                                    by_type_inserted[prev] -= 1
                                    if by_type_inserted[prev] <= 0:
                                        by_type_inserted.pop(prev, None)
                        continue

                    merged = _update_relation_with_review(conn, relation_id, normalized)
                    if merged:
                        collisions_merged_now += 1
                    if item.get("inserted_now") and normalized["relation_type"] == "ACCORDING_TO" and _has_target_norm_key(
                        item["target_norm_key"]
                    ):
                        inserted_according_to_with_target_now += 1

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
        "skipped_according_to_no_target_now": skipped_according_to_no_target_now,
        "inserted_according_to_with_target_now": inserted_according_to_with_target_now,
        "collisions_merged_now": collisions_merged_now,
        "skipped_according_to_no_target_samples": skipped_according_to_no_target_samples,
        "errors": errors,
    }


def _should_persist_relation(relation_type: str, target_norm_key: Any) -> bool:
    if str(relation_type).upper() != "ACCORDING_TO":
        return True
    return _has_target_norm_key(target_norm_key)


def _has_target_norm_key(target_norm_key: Any) -> bool:
    return bool(str(target_norm_key or "").strip())


def _append_skipped_sample(
    samples: list[dict[str, Any]],
    row: sqlite3.Row,
    candidate: RelationCandidate,
    max_items: int = 50,
) -> None:
    if len(samples) >= max_items:
        return
    samples.append(
        {
            "citation_id": row["citation_id"],
            "source_doc_key": row["source_doc_key"],
            "raw_text": row["raw_text"],
            "evidence_snippet": candidate.evidence_snippet,
            "target_norm_key": row["target_norm_key"],
            "relation_type": candidate.relation_type,
            "reason": "ACCORDING_TO requires non-empty target_norm_key",
        }
    )


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
        "c.evidence_snippet, c.evidence_text, c.evidence_kind, "
        "u.unit_number AS source_unit_number, u.text AS source_unit_text, "
        "l.link_id, l.target_norm_key, l.resolution_status "
        "FROM citations c JOIN citation_links l ON l.citation_id = c.citation_id "
        "LEFT JOIN units u ON u.id = CAST(c.source_unit_id AS INTEGER) "
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
    source_unit_text = str(row["source_unit_text"] or "").strip()
    if source_unit_text:
        return source_unit_text
    if source_unit_id and source_unit_id in structured_units:
        text = structured_units[source_unit_id].text.strip()
        if text:
            return text
    parts = [row["evidence_text"] or "", row["raw_text"] or "", row["evidence_snippet"] or ""]
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
    extract_version: str,
) -> tuple[int | None, bool]:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO relation_extractions (
            citation_id, link_id, source_doc_key, source_unit_id, source_unit_number, source_unit_text, target_norm_key,
            relation_type, direction, scope, scope_detail,
            method, confidence, evidence_snippet, extracted_match_snippet, explanation, created_at, extract_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["citation_id"],
            row["link_id"],
            row["source_doc_key"],
            _safe_int(row["source_unit_id"]),
            row["source_unit_number"],
            row["source_unit_text"],
            row["target_norm_key"],
            candidate.relation_type,
            candidate.direction,
            candidate.scope,
            candidate.scope_detail,
            method,
            candidate.confidence,
            candidate.evidence_snippet,
            candidate.evidence_snippet,
            candidate.explanation[:150],
            now,
            extract_version,
        ),
    )
    if cur.rowcount:
        relation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return relation_id, True
    existing = conn.execute(
        """
        SELECT relation_id
        FROM relation_extractions
        WHERE citation_id = ? AND link_id = ? AND source_unit_id = ? AND relation_type = ?
          AND scope = ? AND COALESCE(scope_detail, '') = COALESCE(?, '')
          AND method = ?
          AND extract_version = ?
        """,
        (
            row["citation_id"],
            row["link_id"],
            _safe_int(row["source_unit_id"]),
            candidate.relation_type,
            candidate.scope,
            candidate.scope_detail,
            method,
            extract_version,
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
) -> bool:
    source = conn.execute(
        """
        SELECT relation_id, citation_id, link_id, method,
               relation_type, scope, scope_detail
        FROM relation_extractions
        WHERE relation_id = ?
        """,
        (relation_id,),
    ).fetchone()
    if source is None:
        return False

    existing = conn.execute(
        """
        SELECT relation_id, confidence, explanation
        FROM relation_extractions
        WHERE citation_id = ?
          AND link_id = ?
          AND relation_type = ?
          AND scope = ?
          AND COALESCE(scope_detail, '') = COALESCE(?, '')
          AND method = ?
          AND relation_id <> ?
        LIMIT 1
        """,
        (
            source["citation_id"],
            source["link_id"],
            review["relation_type"],
            review["scope"],
            review["scope_detail"],
            source["method"],
            relation_id,
        ),
    ).fetchone()

    if existing is None:
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
        return False

    merged_confidence = max(float(existing["confidence"] or 0.0), float(review["confidence"] or 0.0))
    llm_explanation = str(review.get("explanation") or "").strip()
    merged_explanation = llm_explanation or str(existing["explanation"] or "")

    conn.execute(
        """
        UPDATE relation_extractions
        SET direction = ?, confidence = ?, explanation = ?
        WHERE relation_id = ?
        """,
        (
            review["direction"],
            merged_confidence,
            merged_explanation,
            existing["relation_id"],
        ),
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO relation_llm_reviews (
            relation_id, llm_model, prompt_version,
            relation_type, direction, scope, scope_detail,
            llm_confidence, explanation, created_at
        )
        SELECT ?, llm_model, prompt_version,
               relation_type, direction, scope, scope_detail,
               llm_confidence, explanation, created_at
        FROM relation_llm_reviews
        WHERE relation_id = ?
        """,
        (existing["relation_id"], relation_id),
    )
    conn.execute("DELETE FROM relation_llm_reviews WHERE relation_id = ?", (relation_id,))
    deleted = _delete_relation_extraction(conn, relation_id)
    return deleted


def _delete_relation_extraction(conn: sqlite3.Connection, relation_id: int) -> bool:
    cur = conn.execute("DELETE FROM relation_extractions WHERE relation_id = ?", (relation_id,))
    return bool(cur.rowcount)


def _safe_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1
