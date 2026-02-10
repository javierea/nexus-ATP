import sqlite3
from pathlib import Path

from rg_atp_pipeline.services import citations_service
from rg_atp_pipeline.services.citations_service import _normalize_review, parse_bool, run_citations
from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def test_citations_service_resolves_and_creates_placeholders(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-001"
    raw_text = (
        "Conforme Ley 83-F y Dec. Ley 2444/62, "
        "además de la Ley 9999-Z."
    )
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    norm_id = repo.upsert_norm("LEY-83-F", "LEY")
    repo.add_alias(norm_id, "Dec. Ley 2444/62", alias_kind="CITATION", confidence=0.9)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert summary["citations_inserted_now"] >= 2

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        links = conn.execute(
            "SELECT resolution_status, target_norm_key FROM citation_links"
        ).fetchall()
        statuses = {row["resolution_status"] for row in links}
        assert "RESOLVED" in statuses
        assert "PLACEHOLDER_CREATED" in statuses
        placeholder = conn.execute(
            "SELECT norm_key FROM norms WHERE norm_key = 'LEY-9999-Z'"
        ).fetchone()
        assert placeholder is not None


def test_citations_service_dedupes_duplicate_candidates(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-002"
    raw_text = "Ley 83-F. Ley 83-F."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=False,
    )

    assert summary["citations_inserted_now"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        citations_count = conn.execute(
            "SELECT COUNT(*) AS total FROM citations"
        ).fetchone()["total"]
        links_count = conn.execute(
            "SELECT COUNT(*) AS total FROM citation_links"
        ).fetchone()["total"]
        assert citations_count == 1
        assert links_count == 1


def test_citations_service_placeholder_fk(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-003"
    raw_text = "Según Ley 1234-A."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert summary["placeholders_created_now"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT l.target_norm_id, l.target_norm_key, n.norm_key
            FROM citation_links l
            JOIN norms n ON n.norm_id = l.target_norm_id
            """
        ).fetchone()
        assert row is not None
        assert row["target_norm_key"] == "LEY-1234-A"
        assert row["norm_key"] == "LEY-1234-A"


def test_citations_service_runs_twice_without_lock(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-004"
    raw_text = "Ley 83-F y Dec. Ley 2444/62."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    norm_id = repo.upsert_norm("LEY-83-F", "LEY")
    repo.add_alias(norm_id, "Dec. Ley 2444/62", alias_kind="CITATION", confidence=0.9)

    summaries = []
    for _ in range(2):
        summaries.append(
            run_citations(
                db_path=db_path,
                data_dir=data_dir,
                doc_keys=[doc_key],
                limit_docs=None,
                llm_mode="off",
                min_confidence=0.7,
                create_placeholders=True,
            )
        )

    assert summaries[0]["citations_inserted_now"] >= 1


def test_citations_service_streamlit_rerun(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-005"
    raw_text = "Según Ley 1234-A."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    first = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )
    second = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    assert first["docs_processed"] == 1
    assert second["docs_processed"] == 1
    assert second["citations_inserted_now"] == 0
    assert second["links_inserted_now"] == 0
    assert second["links_updated_now"] == 0
    assert second["placeholders_created_now"] == 0
    assert second["reviews_inserted_now"] == 0
    assert second["rejected_now"] == 0
    assert second["links_status_totals"] == first["links_status_totals"]


def test_citations_summary_matches_effective_changes(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-006"
    raw_text = "Ley 83-F y Ley 9999-Z."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    repo.upsert_norm("LEY-83-F", "LEY")

    first = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )
    second = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        counts = citations_service._count_links_by_status(  # noqa: SLF001
            conn,
            citations_service.logging.getLogger("test.citations"),
        )

    assert first["links_inserted_now"] == 2
    assert first["links_updated_now"] == 0
    assert second["links_inserted_now"] == 0
    assert second["links_updated_now"] == 0
    assert counts.get("RESOLVED", 0) == 1
    assert counts.get("PLACEHOLDER_CREATED", 0) == 1
    assert first["links_status_totals"] == counts
    assert second["links_status_totals"] == counts


def test_citations_idempotent_single_link_per_citation(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-007"
    raw_text = "Ley 83-F y Ley 1234-A."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    repo = NormsRepository(db_path)
    repo.upsert_norm("LEY-83-F", "LEY")

    run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )
    run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.7,
        create_placeholders=True,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_links,
                COUNT(DISTINCT citation_id) AS distinct_citations
            FROM citation_links
            """
        ).fetchone()

    assert row["total_links"] == row["distinct_citations"]


def test_citations_rejected_only_from_llm_verify(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-008"
    raw_text = "Ley 83-F."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    def fake_verify_candidates(*_args, **_kwargs):
        return [
            {
                "candidate_id": "1",
                "is_reference": False,
                "norm_type": "OTRO",
                "normalized_key": None,
                "confidence": 0.99,
                "explanation": "not a legal citation",
            }
        ]

    monkeypatch.setattr(citations_service, "verify_candidates", fake_verify_candidates)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.7,
        create_placeholders=False,
        llm_gate_regex_threshold=1.0,
    )

    assert summary["rejected_now"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM citation_links WHERE resolution_status = 'REJECTED'"
        ).fetchone()

    assert row["total"] == 1


def test_citations_llm_off_never_creates_rejected(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-009"
    raw_text = "Ley 83-F."
    (raw_text_dir / f"{doc_key}.txt").write_text(raw_text, encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="off",
        min_confidence=0.95,
        create_placeholders=False,
    )

    assert summary["rejected_now"] == 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) FROM citation_links WHERE resolution_status = 'REJECTED'"
        ).fetchone()

    assert row[0] == 0


def test_normalize_review_forces_not_reference_on_high_confidence_negative_explanation():
    review = _normalize_review(
        {
            "candidate_id": "1",
            "is_reference": True,
            "norm_type": "LEY",
            "normalized_key": "LEY-83-F",
            "confidence": 0.99,
            "explanation": "Not a reference",
        }
    )

    assert review is not None
    assert review["is_reference"] is False


def test_hardening_explanation_with_string_values():
    review = _normalize_review(
        {
            "candidate_id": "1",
            "is_reference": "true",
            "norm_type": "LEY",
            "normalized_key": "LEY-83-F",
            "confidence": "0.99",
            "explanation": "Not a reference",
        }
    )

    assert review is not None
    assert review["is_reference"] is False
    assert review["confidence"] == 0.99


def test_parse_bool():
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("no") is False
    assert parse_bool("n") is False

    assert parse_bool("true") is True
    assert parse_bool("1") is True
    assert parse_bool("yes") is True
    assert parse_bool("si") is True
    assert parse_bool("sí") is True
    assert parse_bool("y") is True

    assert parse_bool(True) is True
    assert parse_bool(False) is False

    assert parse_bool(0) is False
    assert parse_bool(1) is True


def test_normalize_review_forces_not_reference_on_negative_explanation_even_low_confidence():
    review = _normalize_review(
        {
            "candidate_id": "1",
            "is_reference": True,
            "norm_type": "LEY",
            "normalized_key": "LEY-83-F",
            "confidence": 0.2,
            "explanation": "Not a reference",
        }
    )

    assert review is not None
    assert review["is_reference"] is False


def test_verify_pipeline_rejects_when_normalized_review_sets_not_reference(
    monkeypatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_text_dir = data_dir / "raw_text"
    raw_text_dir.mkdir(parents=True)
    doc_key = "RG-2024-010"
    (raw_text_dir / f"{doc_key}.txt").write_text("Ley 83-F.", encoding="utf-8")

    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    def fake_verify_candidates(*_args, **_kwargs):
        return [
            {
                "candidate_id": "1",
                "is_reference": True,
                "norm_type": "LEY",
                "normalized_key": "LEY-83-F",
                "confidence": 0.99,
                "explanation": "Not a reference",
            }
        ]

    monkeypatch.setattr(citations_service, "verify_candidates", fake_verify_candidates)

    summary = run_citations(
        db_path=db_path,
        data_dir=data_dir,
        doc_keys=[doc_key],
        limit_docs=None,
        llm_mode="verify",
        min_confidence=0.7,
        create_placeholders=False,
        llm_gate_regex_threshold=1.0,
    )

    assert summary["rejected_now"] == 1


def test_normalize_rejected_links_semantics_reclassifies_non_negative_reviews(tmp_path: Path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        now = "2026-01-01T00:00:00+00:00"
        citations = [
            ("RG-TEST", "", "", "Ley 1", "LEY", None, "snippet 1", 0.8, now),
            ("RG-TEST", "", "", "Ley 2", "LEY", None, "snippet 2", 0.8, now),
            ("RG-TEST", "", "", "Ley 3", "LEY", None, "snippet 3", 0.8, now),
        ]
        conn.executemany(
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
            """,
            citations,
        )
        conn.executemany(
            """
            INSERT INTO citation_links (
                citation_id,
                target_norm_id,
                target_norm_key,
                resolution_status,
                resolution_confidence,
                created_at
            )
            VALUES (?, NULL, NULL, 'REJECTED', 0.95, ?)
            """,
            [
                (1, now),
                (2, now),
                (3, now),
            ],
        )
        conn.executemany(
            """
            INSERT INTO citation_llm_reviews (
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
            VALUES (?, 'model', ?, ?, 'LEY', NULL, 0.9, 'review', ?)
            """,
            [
                (1, 'v1', 0, "2026-01-01T00:00:00+00:00"),
                (2, 'v1', 1, "2026-01-01T00:00:00+00:00"),
            ],
        )
        conn.commit()

    summary = citations_service.normalize_rejected_links_semantics(db_path)
    assert summary["updated_rows"] == 2

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT citation_id, resolution_status FROM citation_links ORDER BY citation_id"
        ).fetchall()

    assert [row["resolution_status"] for row in rows] == [
        "REJECTED",
        "UNRESOLVED",
        "UNRESOLVED",
    ]

    second = citations_service.normalize_rejected_links_semantics(db_path)
    assert second["updated_rows"] == 0
