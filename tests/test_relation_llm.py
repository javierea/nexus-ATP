from rg_atp_pipeline.services.relation_llm import _build_prompt


def test_reltype_v2_prompt_blocks_internal_references() -> None:
    prompt = _build_prompt([], prompt_version="reltype-v2")
    assert "NO clasificar como ACCORDING_TO" in prompt
    assert "No interpretar referencias intra-norma" in prompt


def test_reltype_v1_prompt_keeps_base_instructions() -> None:
    prompt = _build_prompt([], prompt_version="reltype-v1")
    assert "NO clasificar como ACCORDING_TO" not in prompt
