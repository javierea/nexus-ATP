from rg_atp_pipeline.audit_compendio import detect_references_in_text


def test_detects_new_res_format() -> None:
    text = "Se dicta la RES-2024-39-20-1 en la provincia."
    refs = detect_references_in_text(text, page_number=1)
    assert any(ref.doc_key_normalized == "RES-2024-39-20-1" for ref in refs)
    ref = next(ref for ref in refs if ref.doc_key_normalized == "RES-2024-39-20-1")
    assert ref.year == 2024
    assert ref.number == 39
    assert ref.confidence == 1.0


def test_detects_new_res_with_spaces() -> None:
    text = "Se dicta la RES 2025 43 20 1 para el ejercicio."
    refs = detect_references_in_text(text, page_number=1)
    assert any(ref.doc_key_normalized == "RES-2025-43-20-1" for ref in refs)


def test_detects_rg_old_number() -> None:
    text = "Según la RG N° 2172, corresponde..."
    refs = detect_references_in_text(text, page_number=1)
    assert any(ref.doc_key_normalized == "OLD-2172" for ref in refs)
    ref = next(ref for ref in refs if ref.doc_key_normalized == "OLD-2172")
    assert ref.confidence < 1.0
