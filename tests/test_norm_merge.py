import sqlite3
from pathlib import Path

from rg_atp_pipeline.services.norm_merge import MERGE_TRACE_ALIAS_KIND, merge_norm
from rg_atp_pipeline.storage.migrations import ensure_schema
from rg_atp_pipeline.storage.norms_repo import NormsRepository


def _seed_merge_case(db_path: Path) -> tuple[int, int]:
    ensure_schema(db_path)
    repo = NormsRepository(db_path)

    to_id = repo.upsert_norm(
        norm_key="LEY-83-F",
        norm_type="LEY",
        jurisdiction="CHACO",
        year=2024,
        number="83",
        suffix="F",
        title="Ley canónica",
    )
    from_id = repo.upsert_norm(
        norm_key="UNK-D3BE5096",
        norm_type="UNK",
        title="Placeholder",
    )

    repo.add_alias(from_id, "Código Tributario viejo", alias_kind="OTHER")
    repo.add_alias(from_id, "Ley 83 F", alias_kind="ALTERNATIVE")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        citation_id_1 = conn.execute(
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
            ("doc-1", "u-1", "ARTICLE", "UNK ref", "LEY", "UNK-D3BE5096", "snip", 0.9, "2025-01-01T00:00:00+00:00"),
        ).lastrowid
        citation_id_2 = conn.execute(
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
            ("doc-2", "u-2", "ARTICLE", "UNK ref 2", "LEY", "UNK-D3BE5096", "snip", 0.95, "2025-01-01T00:00:00+00:00"),
        ).lastrowid

        link_id_1 = conn.execute(
            """
            INSERT INTO citation_links (
                citation_id,
                target_norm_id,
                target_norm_key,
                resolution_status,
                resolution_confidence,
                created_at
            )
            VALUES (?, ?, ?, 'RESOLVED', 0.9, '2025-01-01T00:00:00+00:00')
            """,
            (citation_id_1, from_id, "UNK-D3BE5096"),
        ).lastrowid
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
            VALUES (?, NULL, ?, 'PLACEHOLDER_CREATED', 0.8, '2025-01-01T00:00:00+00:00')
            """,
            (citation_id_2, "UNK-D3BE5096"),
        )

        conn.execute(
            """
            INSERT INTO relation_extractions (
                citation_id,
                link_id,
                source_doc_key,
                target_norm_key,
                relation_type,
                direction,
                scope,
                scope_detail,
                method,
                confidence,
                evidence_snippet,
                explanation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation_id_1,
                link_id_1,
                "doc-1",
                "UNK-D3BE5096",
                "MODIFICA",
                "OUT",
                "ARTICLE",
                "1",
                "RULE",
                0.7,
                "snippet",
                "explanation",
                "2025-01-01T00:00:00+00:00",
            ),
        )

        source_id = conn.execute(
            """
            INSERT INTO norm_sources (
                norm_id,
                source_kind,
                source_method,
                url,
                is_authoritative,
                notes,
                created_at,
                updated_at
            )
            VALUES (?, 'CONSOLIDATED_CURRENT', 'MANUAL_UPLOAD', 'https://example.com', 1, 'n', '2025-01-01T00:00:00+00:00', '2025-01-01T00:00:00+00:00')
            """,
            (from_id,),
        ).lastrowid

        conn.execute(
            """
            INSERT INTO norm_source_versions (
                source_id,
                sha256,
                downloaded_at,
                pdf_path,
                file_size_bytes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, "abc123", "2025-01-01T00:00:00+00:00", "raw/manual/test.pdf", 123),
        )

    return from_id, to_id


def test_merge_norm_dry_run_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    _seed_merge_case(db_path)

    summary = merge_norm(
        db_path=db_path,
        from_norm_key="UNK-D3BE5096",
        to_norm_key="LEY-83-F",
        apply=False,
    )

    assert summary["mode"] == "dry-run"
    assert summary["rows_affected"]["citation_links"] == 2
    assert summary["rows_affected"]["relation_extractions"] == 1
    assert summary["rows_affected"]["norm_aliases"] == 2
    assert summary["rows_affected"]["norm_sources"] == 1
    assert summary["rows_affected"]["norm_source_versions"] == 1
    assert summary["errors"] == []


def test_merge_norm_apply_moves_references_and_aliases(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rg_atp.sqlite"
    from_id, to_id = _seed_merge_case(db_path)

    summary = merge_norm(
        db_path=db_path,
        from_norm_key="UNK-D3BE5096",
        to_norm_key="LEY-83-F",
        apply=True,
    )

    assert summary["mode"] == "apply"
    assert summary["rows_affected"]["citation_links"] == 2
    assert summary["rows_affected"]["relation_extractions"] == 1
    assert summary["errors"] == []

    with sqlite3.connect(db_path) as conn:
        unresolved_refs = conn.execute(
            """
            SELECT COUNT(*)
            FROM citation_links
            WHERE target_norm_key = 'UNK-D3BE5096'
               OR target_norm_id = ?
            """,
            (from_id,),
        ).fetchone()[0]
        assert unresolved_refs == 0

        moved_refs = conn.execute(
            """
            SELECT COUNT(*)
            FROM citation_links
            WHERE target_norm_key = 'LEY-83-F'
              AND target_norm_id = ?
            """,
            (to_id,),
        ).fetchone()[0]
        assert moved_refs == 2

        relation_targets = conn.execute(
            "SELECT DISTINCT target_norm_key FROM relation_extractions"
        ).fetchall()
        assert [row[0] for row in relation_targets] == ["LEY-83-F"]

        source_aliases_remaining = conn.execute(
            "SELECT COUNT(*) FROM norm_aliases WHERE norm_id = ?",
            (from_id,),
        ).fetchone()[0]
        assert source_aliases_remaining == 0

        target_aliases = conn.execute(
            "SELECT alias_text, alias_kind FROM norm_aliases WHERE norm_id = ?",
            (to_id,),
        ).fetchall()
        alias_pairs = {(row[0], row[1]) for row in target_aliases}
        assert ("Código Tributario viejo", "OTHER") in alias_pairs
        assert ("Ley 83 F", "ALTERNATIVE") in alias_pairs
        assert ("UNK-D3BE5096", MERGE_TRACE_ALIAS_KIND) in alias_pairs

        source_norm_exists = conn.execute(
            "SELECT COUNT(*) FROM norms WHERE norm_key = 'UNK-D3BE5096'"
        ).fetchone()[0]
        assert source_norm_exists == 1

        moved_sources = conn.execute(
            "SELECT COUNT(*) FROM norm_sources WHERE norm_id = ?",
            (to_id,),
        ).fetchone()[0]
        assert moved_sources == 1
