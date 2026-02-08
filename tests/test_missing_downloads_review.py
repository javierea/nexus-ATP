from rg_atp_pipeline.llm_review import MissingDownloadReview, filter_missing_downloads_atp


def test_filter_missing_downloads_atp_filters_by_verdict() -> None:
    reviews = [
        MissingDownloadReview(
            doc_key="RES-2024-1-20-1",
            verdict="ATP_MISSING",
            org_guess="ATP Chaco",
            confidence=0.9,
            reason="ok",
            status="PENDING",
            url="http://example.com",
            last_checked_at=None,
            last_downloaded_at=None,
        ),
        MissingDownloadReview(
            doc_key="OLD-2",
            verdict="DETECTION_ERROR",
            org_guess="Unknown",
            confidence=0.99,
            reason="miles",
            status="PENDING",
            url="http://example.com",
            last_checked_at=None,
            last_downloaded_at=None,
        ),
        MissingDownloadReview(
            doc_key="OLD-3",
            verdict="OTHER_ORG",
            org_guess="AFIP/ARCA",
            confidence=0.95,
            reason="afip",
            status="PENDING",
            url="http://example.com",
            last_checked_at=None,
            last_downloaded_at=None,
        ),
    ]
    filtered = filter_missing_downloads_atp(reviews, confidence_threshold=0.8)
    assert [item.doc_key for item in filtered] == ["RES-2024-1-20-1"]


def test_filter_missing_downloads_atp_applies_threshold() -> None:
    reviews = [
        MissingDownloadReview(
            doc_key="RES-2024-4-20-1",
            verdict="ATP_MISSING",
            org_guess="ATP Chaco",
            confidence=0.79,
            reason="baja",
            status="PENDING",
            url="http://example.com",
            last_checked_at=None,
            last_downloaded_at=None,
        ),
        MissingDownloadReview(
            doc_key="RES-2024-5-20-1",
            verdict="ATP_MISSING",
            org_guess="ATP Chaco",
            confidence=0.85,
            reason="alta",
            status="PENDING",
            url="http://example.com",
            last_checked_at=None,
            last_downloaded_at=None,
        ),
    ]
    filtered = filter_missing_downloads_atp(reviews, confidence_threshold=0.8)
    assert [item.doc_key for item in filtered] == ["RES-2024-5-20-1"]
