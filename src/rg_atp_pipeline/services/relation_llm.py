"""LLM verification for relation type candidates using Ollama."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests


def verify_relation_candidates(
    candidates: list[dict[str, Any]],
    model: str,
    base_url: str,
    prompt_version: str,
    timeout_sec: int,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Verify relation candidates with Ollama and return parsed JSON output."""
    prompt = _build_prompt(candidates, prompt_version)
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/api/generate"):
        endpoint = f"{endpoint}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": False}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            parsed = _parse_response(data.get("response", ""))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                return parsed["items"]
            return []
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if last_error:
        raise last_error
    return []


def _build_prompt(candidates: list[dict[str, Any]], prompt_version: str) -> str:
    instructions = [
        "Responde SOLO JSON válido (array).",
        "No inventar artículos ni números.",
        "Si no es explícito, scope_detail=null.",
        "Si no está claro, relation_type=UNKNOWN con baja confidence.",
    ]
    if str(prompt_version).strip().lower() == "reltype-v2":
        instructions.extend(
            [
                "Si solo hay referencia interna a artículo/inciso/anexo sin otra norma externa, NO clasificar como ACCORDING_TO.",
                "No interpretar referencias intra-norma como relación entre normas; usa relation_type=UNKNOWN o confidence baja.",
            ]
        )

    payload = {
        "version": prompt_version,
        "instructions": instructions,
        "input": candidates,
        "output_schema": {
            "candidate_id": "string",
            "relation_type": "REPEALS|MODIFIES|SUBSTITUTES|INCORPORATES|REGULATES|COMPLEMENTS|ACCORDING_TO|UNKNOWN",
            "direction": "SOURCE_TO_TARGET|UNKNOWN",
            "scope": "WHOLE_NORM|ARTICLE|ANNEX|UNKNOWN",
            "scope_detail": "string|null",
            "confidence": "0..1",
            "explanation": "max 20 palabras",
        },
    }
    return (
        "Eres un clasificador de relaciones normativas. "
        "Valida candidatos con criterio conservador. Devuelve SOLO JSON.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_response(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        return []

