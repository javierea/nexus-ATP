from rg_atp_pipeline.services.relation_extractor import extract_relation_candidates


def test_relation_extractor_repeals_whole_norm() -> None:
    text = "Derógase la RG ATP 123/2020."
    items = extract_relation_candidates(text)
    assert any(
        item.relation_type == "REPEALS" and item.scope == "WHOLE_NORM"
        for item in items
    )


def test_relation_extractor_modifies_article_scope() -> None:
    text = "Modifícase el art. 12 de la Ley 1234-A."
    items = extract_relation_candidates(text)
    assert any(
        item.relation_type == "MODIFIES"
        and item.scope == "ARTICLE"
        and item.scope_detail == "ART_12"
        for item in items
    )


def test_relation_extractor_substitutes_annex_scope() -> None:
    text = "Sustitúyese el Anexo I de la RG ATP 50/2019."
    items = extract_relation_candidates(text)
    assert any(
        item.relation_type == "SUBSTITUTES"
        and item.scope == "ANNEX"
        and item.scope_detail == "ANEXO_I"
        for item in items
    )


def test_relation_extractor_according_to() -> None:
    text = "Según Ley 83-F se aplican los plazos previstos."
    items = extract_relation_candidates(text)
    assert any(
        item.relation_type == "ACCORDING_TO" and item.direction == "UNKNOWN"
        for item in items
    )


def test_relation_extractor_multiple_modifies_matches() -> None:
    text = "Modifícase el art. 12 de la Ley 1234-A. Modifícase el art. 15 de la Ley 1234-A."
    items = extract_relation_candidates(text)
    modifies = [
        item
        for item in items
        if item.relation_type == "MODIFIES" and item.scope == "ARTICLE"
    ]
    details = {item.scope_detail for item in modifies}
    assert "ART_12" in details
    assert "ART_15" in details
    assert len(details) == 2
