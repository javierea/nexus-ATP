import json
import sqlite3
from pathlib import Path

import responses

from rg_atp_pipeline.config import Config, OldRange, RetryPolicy, TextQualityConfig
from rg_atp_pipeline.fetcher import FetchOptions, run_fetch
from rg_atp_pipeline.paths import data_dir, state_path
from rg_atp_pipeline.state import default_state, save_state
from rg_atp_pipeline.storage_sqlite import DocumentStore


def _config(base_url_old: str) -> Config:
    return Config(
        base_url_new="https://example.com/new",
        base_url_old=base_url_old,
        rate_limit_rps=1000,
        user_agent="rg_atp_pipeline-test",
        years=[2026],
        max_n_by_year={"2026": 1},
        old_range=OldRange(start=1, end=1),
        old_min_number=1,
        old_year_start=2023,
        old_year_end=2023,
        verify_last_k=5,
        request_timeout_sec=5,
        retry=RetryPolicy(max_attempts=1, backoff_sec=0),
        text_quality=TextQualityConfig(
            min_chars_total=1,
            min_chars_per_page=1,
            min_alpha_ratio=0.0,
        ),
    )


def _setup_state(tmp_path: Path) -> DocumentStore:
    save_state(default_state(), state_path())
    store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
    store.initialize()
    return store


def _fetch_options() -> FetchOptions:
    return FetchOptions(
        mode="old",
        year=None,
        n_start=None,
        n_end=None,
        old_start=1,
        old_end=1,
        dry_run=False,
        max_downloads=None,
        skip_existing=False,
    )


@responses.activate
def test_head_404_marks_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    cfg = _config("https://example.com/old")
    store = _setup_state(tmp_path)

    responses.add(responses.HEAD, "https://example.com/old/1.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/1-2023.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/1-23.pdf", status=404)

    summary = run_fetch(cfg, default_state(), store, data_dir(), _fetch_options(), _logger())
    assert summary.missing == 1

    with sqlite3.connect(data_dir() / "state" / "rg_atp.sqlite") as conn:
        row = conn.execute("SELECT status, http_status FROM documents WHERE doc_key = ?", ("OLD-1",)).fetchone()
    assert row == ("MISSING", 404)


@responses.activate
def test_get_200_saves_versioned_pdf(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    cfg = _config("https://example.com/old")
    store = _setup_state(tmp_path)
    content = b"pdf-content-v1"

    responses.add(responses.HEAD, "https://example.com/old/1.pdf", status=200)
    responses.add(responses.GET, "https://example.com/old/1.pdf", status=200, body=content)

    summary = run_fetch(cfg, default_state(), store, data_dir(), _fetch_options(), _logger())
    assert summary.downloaded == 1

    sha = _sha(content)
    pdf_path = data_dir() / "raw_pdfs" / f"OLD-1__{sha}.pdf"
    assert pdf_path.exists()
    _assert_latest_pointer(pdf_path)


@responses.activate
def test_changed_hash_creates_new_version(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    cfg = _config("https://example.com/old")
    store = _setup_state(tmp_path)

    first = b"pdf-content-v1"
    second = b"pdf-content-v2"

    responses.add(responses.HEAD, "https://example.com/old/1.pdf", status=200)
    responses.add(responses.GET, "https://example.com/old/1.pdf", status=200, body=first)
    run_fetch(cfg, default_state(), store, data_dir(), _fetch_options(), _logger())

    responses.reset()
    responses.add(responses.HEAD, "https://example.com/old/1.pdf", status=200)
    responses.add(responses.GET, "https://example.com/old/1.pdf", status=200, body=second)
    summary = run_fetch(cfg, default_state(), store, data_dir(), _fetch_options(), _logger())

    assert summary.changed_hash == 1
    first_sha = _sha(first)
    second_sha = _sha(second)
    first_path = data_dir() / "raw_pdfs" / f"OLD-1__{first_sha}.pdf"
    second_path = data_dir() / "raw_pdfs" / f"OLD-1__{second_sha}.pdf"
    assert first_path.exists()
    assert second_path.exists()
    _assert_latest_pointer(second_path)


def _assert_latest_pointer(target: Path) -> None:
    latest_dir = data_dir() / "raw_pdfs" / "latest"
    symlink = latest_dir / "OLD-1.pdf"
    pointer = latest_dir / "OLD-1.json"
    if symlink.exists() or symlink.is_symlink():
        assert symlink.resolve() == target
    else:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        assert payload["latest_pdf_path"] == str(target)


def _sha(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def _logger():
    import logging

    logger = logging.getLogger("rg_atp_pipeline.tests")
    logger.addHandler(logging.NullHandler())
    return logger


@responses.activate
def test_old_suffix_year_cuts_off_higher_years(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RG_ATP_PIPELINE_ROOT", str(tmp_path))
    cfg = Config(
        base_url_new="https://example.com/new",
        base_url_old="https://example.com/old",
        rate_limit_rps=1000,
        user_agent="rg_atp_pipeline-test",
        years=[2026],
        max_n_by_year={"2026": 1},
        old_range=OldRange(start=1, end=2),
        old_min_number=1,
        old_year_start=2014,
        old_year_end=2016,
        verify_last_k=5,
        request_timeout_sec=5,
        retry=RetryPolicy(max_attempts=1, backoff_sec=0),
        text_quality=TextQualityConfig(
            min_chars_total=1,
            min_chars_per_page=1,
            min_alpha_ratio=0.0,
        ),
    )
    store = _setup_state(tmp_path)

    responses.add(responses.HEAD, "https://example.com/old/2.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-2016.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-16.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-2015.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-15.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-2014.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/2-14.pdf", status=200)
    responses.add(
        responses.GET, "https://example.com/old/2-14.pdf", status=200, body=b"pdf-content"
    )

    responses.add(responses.HEAD, "https://example.com/old/1.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/1-2014.pdf", status=404)
    responses.add(responses.HEAD, "https://example.com/old/1-14.pdf", status=404)

    options = FetchOptions(
        mode="old",
        year=None,
        n_start=None,
        n_end=None,
        old_start=1,
        old_end=2,
        dry_run=False,
        max_downloads=None,
        skip_existing=False,
    )

    summary = run_fetch(cfg, default_state(), store, data_dir(), options, _logger())
    assert summary.downloaded == 1
    assert summary.missing == 1
