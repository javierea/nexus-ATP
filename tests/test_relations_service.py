import sqlite3
import json
from pathlib import Path

from rg_atp_pipeline.services.relation_extractor import RelationCandidate
from rg_atp_pipeline.services.relations_service import _insert_relation_extraction, run_relations
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_relations_service_inserts_regex_relations(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-900",
                "1",
                "ARTICLE",
                "Derógase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase la Ley 83-F de forma integral",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'PLACEHOLDER_CREATED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-900"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 1
    assert summary["by_type"].get("REPEALS") == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT relation_type, scope, method FROM relation_extractions"
        ).fetchone()
    assert row is not None
    assert row["relation_type"] == "REPEALS"
    assert row["scope"] == "WHOLE_NORM"
    assert row["method"] == "REGEX"


def test_relations_service_uses_structured_unit_text(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    structured_dir = data_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    doc_key = "RG-2024-901"
    unit_id = "U-1"
    structured_payload = {
        "doc_key": doc_key,
        "units": [
            {
                "unit_id": unit_id,
                "unit_type": "ARTICLE",
                "text": "Texto inicial. Derógase la Ley 83-F. Texto final.",
            }
        ],
    }
    (structured_dir / f"{doc_key}.json").write_text(
        json.dumps(structured_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_key,
                unit_id,
                "ARTICLE",
                "Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Ley 83-F",
                0.7,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 1
    assert summary["by_type"].get("REPEALS") == 1


def test_relations_service_ignores_unresolved_links(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-902",
                "1",
                "ARTICLE",
                "Derógase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase la Ley 83-F",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'UNRESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.2, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-902"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["links_seen"] == 0
    assert summary["relations_inserted"] == 0


def test_relations_service_verify_mode_gates_candidates_and_reports_metrics(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-903",
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Modifícase la Ley 83-F",
                0.7,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    def fake_verify(payload, **_kwargs):
        assert len(payload) == 1
        return [
            {
                "candidate_id": payload[0]["candidate_id"],
                "relation_type": "MODIFIES",
                "direction": "OUTGOING",
                "scope": "ARTICLE",
                "scope_detail": "1",
                "confidence": 0.88,
                "explanation": "LLM verified",
            }
        ]

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        fake_verify,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-903"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["total_candidates_detected"] == 1
    assert summary["candidates_inserted_now"] == 1
    assert summary["gated_count"] == 1
    assert summary["batches_sent"] == 1
    assert summary["llm_verified"] == 1
    assert summary["by_type_detected"]["MODIFIES"] == 1
    assert summary["by_type_inserted"]["MODIFIES"] == 1


def test_relations_service_skips_according_to_without_target_norm_key(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-904",
                "1",
                "ARTICLE",
                "según lo establece el Artículo 16º",
                "LEY",
                None,
                "según lo establece el Artículo 16º",
                0.8,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'PLACEHOLDER_CREATED', ?, ?)
            """,
            (citation_id, "", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="ACCORDING_TO",
                direction="UNKNOWN",
                scope="ARTICLE",
                scope_detail="16",
                confidence=0.9,
                evidence_snippet="según lo establece el Artículo 16º",
                explanation="regex according_to",
            )
        ]

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-904"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 0
    assert summary["skipped_according_to_no_target_now"] == 1
    assert summary["inserted_according_to_with_target_now"] == 0

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
    assert count == 0


def test_relations_service_inserts_according_to_with_target_norm_key(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-905",
                "1",
                "ARTICLE",
                "según Ley 83-F",
                "LEY",
                "LEY-83-F",
                "según Ley 83-F",
                0.8,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="ACCORDING_TO",
                direction="SOURCE_TO_TARGET",
                scope="WHOLE_NORM",
                scope_detail=None,
                confidence=0.9,
                evidence_snippet="según Ley 83-F",
                explanation="regex according_to",
            )
        ]

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-905"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["relations_inserted"] == 1
    assert summary["inserted_according_to_with_target_now"] == 1
    assert summary["skipped_according_to_no_target_now"] == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT relation_type, target_norm_key FROM relation_extractions"
        ).fetchone()
    assert row is not None
    assert row[0] == "ACCORDING_TO"
    assert row[1] == "LEY-83-F"


def test_relations_service_merges_llm_update_when_unique_key_collides(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-906",
                "1",
                "ARTICLE",
                "Derógase y modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase y modifícase la Ley 83-F",
                0.8,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, target_norm_key,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id,
                link_id,
                "RG-2024-906",
                "LEY-83-F",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.75,
                "modifícase",
                "existing relation",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="REPEALS",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Derógase",
                explanation="regex repeals",
            )
        ]

    def fake_verify(payload, **_kwargs):
        return [
            {
                "candidate_id": payload[0]["candidate_id"],
                "relation_type": "MODIFIES",
                "direction": "OUTGOING",
                "scope": "ARTICLE",
                "scope_detail": "1",
                "confidence": 0.91,
                "explanation": "LLM corrected",
            }
        ]

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        fake_verify,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-906"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["llm_verified"] == 1
    assert summary["collisions_merged_now"] == 1

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT relation_type, scope, scope_detail, method, confidence, explanation FROM relation_extractions ORDER BY relation_id"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "MODIFIES"
    assert rows[0][1] == "ARTICLE"
    assert rows[0][2] == "1"
    assert rows[0][3] == "MIXED"
    assert rows[0][4] == 0.91
    assert rows[0][5] == "LLM corrected"


def test_relations_service_upserts_duplicate_regex_relations_and_reuses_relation_for_review(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-908",
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Modifícase la Ley 83-F",
                0.75,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id,
                link_id,
                "RG-2024-908",
                1,
                "1",
                "Modifícase la Ley 83-F",
                "LEY-83-F",
                "relext-v2",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.61,
                "",
                "",
                "",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        existing_relation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase",
                explanation="candidate low",
            ),
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.8,
                evidence_snippet="Modifícase la Ley 83-F completa",
                explanation="candidate better",
            ),
        ]

    def fake_verify(payload, **_kwargs):
        return [
            {
                "candidate_id": payload[0]["candidate_id"],
                "relation_type": "MODIFIES",
                "direction": "OUTGOING",
                "scope": "ARTICLE",
                "scope_detail": "1",
                "confidence": 0.9,
                "explanation": "LLM ok",
            }
        ]

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        fake_verify,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-908"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["collisions_merged_now"] == 1
    assert summary["updates_now"] >= 1

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
        review_relation_id = conn.execute(
            "SELECT relation_id FROM relation_llm_reviews ORDER BY review_id DESC LIMIT 1"
        ).fetchone()[0]

    assert count == 1
    assert review_relation_id == existing_relation_id


def test_insert_relation_extraction_handles_upsert_integrity_and_updates_existing_key(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-909",
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "snippet-1",
                0.75,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id_1, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-909",
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-G",
                "LEY",
                "LEY-83-G",
                "snippet-2",
                0.72,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id_2, "LEY-83-G", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_1,
                link_id_1,
                "RG-2024-909",
                1,
                "1",
                "texto base",
                "LEY-83-F",
                "relext-v2",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.8,
                "base",
                "base",
                "existing",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_2,
                link_id_2,
                "RG-2024-909",
                1,
                "1",
                "texto alterno",
                "LEY-83-G",
                "relext-v2",
                "MODIFIES",
                "OUTGOING",
                "ARTICLE",
                "1",
                "MIXED",
                0.6,
                "alt",
                "alt",
                "existing alt",
                "2026-01-01T00:00:00+00:00",
            ),
        )

        row = conn.execute(
            """
            SELECT c.citation_id, c.source_doc_key, c.source_unit_id,
                   c.evidence_snippet, c.evidence_text, c.raw_text,
                   l.link_id, l.target_norm_key,
                   u.unit_number AS source_unit_number, u.text AS source_unit_text
            FROM citations c
            JOIN citation_links l ON l.citation_id = c.citation_id
            LEFT JOIN units u ON u.id = CAST(c.source_unit_id AS INTEGER)
            WHERE c.citation_id = ?
            """,
            (citation_id_2,),
        ).fetchone()

        candidate = RelationCandidate(
            relation_type="MODIFIES",
            direction="OUTGOING",
            scope="ARTICLE",
            scope_detail="1",
            confidence=0.9,
            evidence_snippet="snippet actualizado",
            explanation="nuevo",
        )

        relation_id, inserted, updated = _insert_relation_extraction(
            conn,
            row,
            candidate,
            method="MIXED",
            extract_version="relext-v2",
        )

        assert relation_id is not None
        assert inserted is False
        assert updated is True

        merged = conn.execute(
            """
            SELECT citation_id, link_id, confidence
            FROM relation_extractions
            WHERE source_doc_key = ? AND source_unit_id = ? AND target_norm_key = ?
              AND relation_type = ? AND scope = ? AND scope_detail = ?
              AND method = ? AND extract_version = ?
            """,
            ("RG-2024-909", 1, "LEY-83-G", "MODIFIES", "ARTICLE", "1", "MIXED", "relext-v2"),
        ).fetchone()
        assert merged is not None
        assert merged[0] == citation_id_2
        assert merged[1] == link_id_2
        assert merged[2] == 0.9


def test_relations_service_handles_unparseable_llm_string_response(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-907",
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Modifícase la Ley 83-F",
                0.7,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    def fake_verify(_payload, **_kwargs):
        return "not-json-response"

    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.extract_relation_candidates",
        fake_extract,
    )
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        fake_verify,
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-907"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["gated_count"] == 1
    assert summary["llm_verified"] == 0
    assert summary["parse_error_now"] >= 1
    assert summary["empty_response_now"] == 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        review = conn.execute(
            """
            SELECT status, raw_response, relation_type
            FROM relation_llm_reviews
            ORDER BY review_id DESC
            LIMIT 1
            """
        ).fetchone()

    assert review is not None
    assert review["status"] == "PARSE_ERROR"
    assert "not-json-response" in (review["raw_response"] or "")
    assert review["relation_type"] == "UNKNOWN"


def _seed_verify_relation_case(db_path: Path, doc_key: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_key,
                "1",
                "ARTICLE",
                "Modifícase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Modifícase la Ley 83-F",
                0.7,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()


def test_relations_service_llm_invalid_id_and_missing_result(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-910")

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            ),
            RelationCandidate(
                relation_type="REPEALS",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="2",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate 2",
            ),
        ]

    def fake_verify(_payload, **_kwargs):
        return [{"candidate_id": "LEY-83-F", "relation_type": "MODIFIES"}]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.verify_relation_candidates", fake_verify)

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-910"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["invalid_id_now"] >= 1
    assert summary["missing_result_now"] >= 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT status, raw_item FROM relation_llm_reviews"
        ).fetchall()
    assert rows
    statuses = {row["status"] for row in rows}
    assert "MISSING_RESULT" in statuses


def test_relations_service_llm_empty_response_marks_all_expected(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-911")

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.verify_relation_candidates", lambda *_a, **_k: "")

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-911"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["empty_response_now"] == 1
    with sqlite3.connect(db_path) as conn:
        status = conn.execute("SELECT status FROM relation_llm_reviews LIMIT 1").fetchone()[0]
    assert status == "EMPTY_RESPONSE"


def test_relations_service_llm_invalid_structure_marks_invalid_response(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-912")

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            ),
            RelationCandidate(
                relation_type="REPEALS",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="2",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate 2",
            ),
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        lambda *_a, **_k: [{"candidate_id": None, "relation_type": "MODIFIES"}],
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-912"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["invalid_id_now"] >= 1
    assert summary["missing_result_now"] >= 1


def test_relations_service_verify_all_gates_all_eligible(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-913")

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.95,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    def fake_verify(payload, **_kwargs):
        return [
            {
                "candidate_id": payload[0]["candidate_id"],
                "relation_type": "MODIFIES",
                "direction": "OUTGOING",
                "scope": "ARTICLE",
                "scope_detail": "1",
                "confidence": 0.88,
                "explanation": "LLM verified",
            }
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.verify_relation_candidates", fake_verify)

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-913"],
        limit_docs=None,
        llm_mode="verify_all",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["gated_count"] == summary["candidates_inserted_now"] == 1
    assert summary["ok_reviews_now"] == 1
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_llm_reviews").fetchone()[0]
    assert count == 1


def test_relations_service_invalid_id_persists_raw_item(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-914")

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        lambda *_a, **_k: [{"candidate_id": "LEY-83-F", "foo": "bar"}],
    )

    run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-914"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status, raw_item FROM relation_llm_reviews WHERE status = 'INVALID_ID' LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row["raw_item"]


def test_relations_service_upsert_normalizes_target_norm_key_null_and_empty(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-915",
                "1",
                "ARTICLE",
                "Derógase algo",
                "LEY",
                "",
                "Derógase",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id,
                link_id,
                "RG-2024-915",
                1,
                "1",
                "Derógase",
                None,
                "relext-v2",
                "REPEALS",
                "OUTGOING",
                "WHOLE_NORM",
                "",
                "REGEX",
                0.6,
                "",
                "",
                "",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="REPEALS",
                direction="OUTGOING",
                scope="WHOLE_NORM",
                scope_detail="",
                confidence=0.95,
                evidence_snippet="Derógase",
                explanation="regex",
            )
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-915"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["collisions_merged_now"] == 1

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
        target_norm_key = conn.execute(
            "SELECT target_norm_key FROM relation_extractions LIMIT 1"
        ).fetchone()[0]

    assert count == 1
    assert target_norm_key == ""


def test_relations_service_upsert_normalizes_scope_detail_null_and_empty(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO citations (
                source_doc_key, source_unit_id, source_unit_type, raw_text,
                norm_type_guess, norm_key_candidate, evidence_snippet,
                regex_confidence, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RG-2024-916",
                "1",
                "ARTICLE",
                "Derógase la Ley 83-F",
                "LEY",
                "LEY-83-F",
                "Derógase la Ley 83-F",
                0.95,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        citation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO citation_links (
                citation_id, target_norm_id, target_norm_key, resolution_status,
                resolution_confidence, created_at
            ) VALUES (?, NULL, ?, 'RESOLVED', ?, ?)
            """,
            (citation_id, "LEY-83-F", 0.95, "2026-01-01T00:00:00+00:00"),
        )
        link_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id, link_id, source_doc_key, source_unit_id, source_unit_number,
                source_unit_text, target_norm_key, extract_version,
                relation_type, direction, scope, scope_detail,
                method, confidence, evidence_snippet, extracted_match_snippet,
                explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id,
                link_id,
                "RG-2024-916",
                1,
                "1",
                "Derógase la Ley 83-F",
                "LEY-83-F",
                "relext-v2",
                "REPEALS",
                "OUTGOING",
                "WHOLE_NORM",
                None,
                "REGEX",
                0.6,
                "",
                "",
                "",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    def fake_extract(_text):
        from rg_atp_pipeline.services.relation_extractor import RelationCandidate

        return [
            RelationCandidate(
                relation_type="REPEALS",
                direction="OUTGOING",
                scope="WHOLE_NORM",
                scope_detail="",
                confidence=0.95,
                evidence_snippet="Derógase la Ley 83-F",
                explanation="regex",
            )
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-916"],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.6,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["collisions_merged_now"] == 1

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
        scope_detail = conn.execute(
            "SELECT scope_detail FROM relation_extractions LIMIT 1"
        ).fetchone()[0]

    assert count == 1
    assert scope_detail == ""


def test_relations_service_llm_candidate_id_must_be_literal(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    _seed_verify_relation_case(db_path, "RG-2024-916")

    def fake_extract(_text):
        return [
            RelationCandidate(
                relation_type="MODIFIES",
                direction="OUTGOING",
                scope="ARTICLE",
                scope_detail="1",
                confidence=0.7,
                evidence_snippet="Modifícase la Ley 83-F",
                explanation="regex candidate",
            )
        ]

    monkeypatch.setattr("rg_atp_pipeline.services.relations_service.extract_relation_candidates", fake_extract)
    monkeypatch.setattr(
        "rg_atp_pipeline.services.relations_service.verify_relation_candidates",
        lambda *_a, **_k: [{"candidate_id": "01", "relation_type": "MODIFIES"}],
    )

    summary = run_relations(
        db_path=db_path,
        data_dir=tmp_path,
        doc_keys=["RG-2024-916"],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.9,
        prompt_version="reltype-v1",
        batch_size=5,
    )

    assert summary["invalid_id_now"] >= 1


def test_insert_relation_extraction_handles_legacy_unit_index_collisions(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("DROP INDEX IF EXISTS ux_relation_extractions_unit_target_type")
        conn.execute(
            """
            CREATE UNIQUE INDEX ux_relation_extractions_unit_target_type
            ON relation_extractions(
                source_doc_key,
                source_unit_id,
                COALESCE(target_norm_key, ''),
                relation_type,
                scope,
                COALESCE(scope_detail, '')
            )
            """
        )

        row = {
            "citation_id": 1,
            "link_id": 1,
            "source_doc_key": "RG-LEGACY",
            "source_unit_id": 10,
            "source_unit_number": "10",
            "source_unit_text": "Derógase la Ley X",
            "target_norm_key": "LEY-X",
        }
        candidate = RelationCandidate(
            relation_type="REPEALS",
            direction="SOURCE_TO_TARGET",
            scope="WHOLE_NORM",
            scope_detail=None,
            confidence=0.9,
            evidence_snippet="Derógase",
            explanation="Derogación detectada",
        )

        first_id, inserted_first, updated_first = _insert_relation_extraction(
            conn,
            row=row,
            candidate=candidate,
            method="REGEX",
            extract_version="relext-v2",
        )
        assert inserted_first is True
        assert updated_first is False

        second_id, inserted_second, updated_second = _insert_relation_extraction(
            conn,
            row=row,
            candidate=RelationCandidate(
                relation_type="REPEALS",
                direction="SOURCE_TO_TARGET",
                scope="WHOLE_NORM",
                scope_detail=None,
                confidence=0.8,
                evidence_snippet="Derógase",
                explanation="Derogación detectada",
            ),
            method="MIXED",
            extract_version="relext-v2",
        )

        assert second_id == first_id
        assert inserted_second is False
        assert updated_second is True

        count = conn.execute("SELECT COUNT(*) FROM relation_extractions").fetchone()[0]
        assert count == 1
