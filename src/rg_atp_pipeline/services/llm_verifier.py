"""LLM verification for citation candidates using Ollama."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests


def verify_candidates(
    candidates: list[dict[str, Any]],
    model: str,
    base_url: str,
    prompt_version: str,
    timeout_sec: int,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Verify candidates with Ollama and return parsed JSON output."""
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
            content = data.get("response", "")
            parsed = _parse_response(content)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "items" in parsed:
                items = parsed.get("items")
                if isinstance(items, list):
                    return items
            return []
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if last_error:
        raise last_error
    return []


def _build_prompt(candidates: list[dict[str, Any]], prompt_version: str) -> str:
    payload = {
        "version": prompt_version,
        "instructions": [
            "Responde SOLO con JSON válido (array). Sin texto adicional.",
            "No inventes números de ley/decreto.",
            "No infieras artículos.",
            "Si no está explícito, normalized_key debe ser null.",
        ],
        "input": candidates,
        "output_schema": {
            "candidate_id": "string",
            "is_reference": "true|false",
            "norm_type": "LEY|DECRETO|RG_ATP|RG_CA|OTRO",
            "normalized_key": "string|null",
            "confidence": "0..1",
            "explanation": "max 20 palabras",
        },
    }
    return (
        "Eres un verificador de referencias normativas. "
        "Valida si los candidatos son referencias reales y normaliza. "
        "Devuelve SOLO JSON.\n"
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
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return []
