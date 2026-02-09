from rg_atp_pipeline.services.llm_verifier import verify_candidates


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_verify_candidates_parses_json(monkeypatch):
    def fake_post(url, json, timeout):
        return DummyResponse(
            {
                "response": (
                    "Resultado:\n"
                    '[{"candidate_id": "1", "is_reference": true, '
                    '"norm_type": "LEY", "normalized_key": "LEY-83-F", '
                    '"confidence": 0.92, "explanation": "mencion explicita"}]'
                )
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = verify_candidates(
        [{"candidate_id": "1", "raw_text": "Ley 83-F"}],
        model="qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        prompt_version="citref-v1",
        timeout_sec=5,
    )
    assert result[0]["candidate_id"] == "1"
    assert result[0]["normalized_key"] == "LEY-83-F"
