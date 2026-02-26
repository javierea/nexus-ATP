from rg_atp_pipeline.services.relation_llm import _build_prompt, _parse_response


def test_reltype_v2_prompt_blocks_internal_references() -> None:
    prompt = _build_prompt([], prompt_version="reltype-v2")
    assert "NO clasificar como ACCORDING_TO" in prompt
    assert "No interpretar referencias intra-norma" in prompt


def test_reltype_v1_prompt_keeps_base_instructions() -> None:
    prompt = _build_prompt([], prompt_version="reltype-v1")
    assert "NO clasificar como ACCORDING_TO" not in prompt


def test_prompt_requires_literal_candidate_id_contract() -> None:
    prompt = _build_prompt([], prompt_version="reltype-v1")
    assert "candidate_id EXACTAMENTE igual" in prompt


def test_parse_response_marks_non_empty_unparsable_payload() -> None:
    parsed = _parse_response("not-json-response")
    assert parsed.get("_parse_error") is True


def test_parse_response_keeps_empty_payload_as_empty_list() -> None:
    parsed = _parse_response("   ")
    assert parsed == []
