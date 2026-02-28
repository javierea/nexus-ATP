import sqlite3
from pathlib import Path

from rg_atp_pipeline.services.graphrag_validation import run_graphrag_validation
from rg_atp_pipeline.storage.migrations import ensure_schema


def test_graphrag_validation_reports_readiness(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO citations (citation_id, source_doc_key, source_unit_id, source_unit_type, raw_text, norm_type_guess, norm_key_candidate, evidence_snippet, evidence_unit_id, evidence_text, evidence_kind, regex_confidence, detected_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "RG-7", "1", "ARTICULO", "Ley 83-F", "LEY", "LEY-83-F", "snippet", 1, "evidence", "UNIT", 0.9, "2026-01-01T00:00:00Z", "citext-v2"),
        )
        conn.execute(
            "INSERT INTO relation_extractions (citation_id, link_id, source_doc_key, source_unit_id, source_unit_number, source_unit_text, target_norm_key, relation_type, direction, scope, scope_detail, method, confidence, evidence_snippet, extracted_match_snippet, explanation, created_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, None, "RG-7", 1, "1", "Artículo 1", "LEY-83-F", "REPEALS", "SOURCE_TO_TARGET", "WHOLE_NORM", "", "REGEX", 0.9, "Derógase", "Derógase", "ok", "2026-01-01T00:00:00Z", "relext-v2"),
        )
        conn.execute(
            "INSERT INTO relation_extractions (citation_id, link_id, source_doc_key, source_unit_id, source_unit_number, source_unit_text, target_norm_key, relation_type, direction, scope, scope_detail, method, confidence, evidence_snippet, extracted_match_snippet, explanation, created_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, None, "RG-7", 1, "1", "Artículo 1", "LEY-83-F", "MODIFIES", "SOURCE_TO_TARGET", "ARTICLE", "ART_1", "REGEX", 0.9, "Modifícase", "Modifícase", "ok", "2026-01-01T00:00:00Z", "relext-v2"),
        )
        conn.execute(
            "INSERT INTO relation_extractions (citation_id, link_id, source_doc_key, source_unit_id, source_unit_number, source_unit_text, target_norm_key, relation_type, direction, scope, scope_detail, method, confidence, evidence_snippet, extracted_match_snippet, explanation, created_at, extract_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, None, "RG-7", 1, "1", "Artículo 1", "LEY-83-F", "ACCORDING_TO", "UNKNOWN", "WHOLE_NORM", "", "REGEX", 0.7, "Según", "Según", "ok", "2026-01-01T00:00:00Z", "relext-v2"),
        )
        conn.execute(
            "INSERT INTO intra_norm_relations (source_doc_key, source_unit_id, source_unit_number, target_unit_id, target_unit_number, relation_type, evidence_snippet, confidence, method, extract_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("RG-7", 1, "1", 2, "2", "REFERS_INTERNAL_ARTICLE", "Ver art. 2", 0.75, "REGEX", "relext-v2", "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    summary = run_graphrag_validation(db_path)
    assert summary["kpis"]["total_external_relations"] == 3
    assert summary["kpis"]["total_intra_norm_relations"] == 1
    assert summary["graphrag_ready"] is True
