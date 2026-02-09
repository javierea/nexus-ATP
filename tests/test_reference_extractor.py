from rg_atp_pipeline.services.reference_extractor import extract_candidates


def test_extract_candidates_detects_expected_patterns():
    text = (
        "Se aplica la Ley 3898-F y la Ley 83-F. "
        "Además, el Dec. Ley 2444/62 establece ... "
        "Ver RES-2025-43-20-1 y RG CA 1/2020."
    )
    candidates = extract_candidates(text)
    raw_texts = {candidate.raw_text for candidate in candidates}
    assert "Ley 3898-F" in raw_texts
    assert "Ley 83-F" in raw_texts
    assert "Dec. Ley 2444/62" in raw_texts
    assert "RES-2025-43-20-1" in raw_texts
    assert "RG CA 1/2020" in raw_texts
