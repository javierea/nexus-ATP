"""LLM verification for relation type candidates using Ollama."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests


logger = logging.getLogger("rg_atp_pipeline.relation_llm")


class RelationLLMTransportError(RuntimeError):
    """Error invoking Ollama for Stage 4.1 with transport diagnostics."""

    def __init__(self, message: str, *, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


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
    transport_audit: list[dict[str, Any]] = []
    for attempt in range(max_retries):
        started_at = time.perf_counter()
        response = None
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_sec)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            body_size = len(response.content or b"")
            transport_audit.append(
                {
                    "attempt": attempt + 1,
                    "retry": max(0, attempt),
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "body_size": body_size,
                    "exception_type": None,
                }
            )
            logger.info(
                "Stage 4.1 Ollama request attempt=%s retry=%s status_code=%s elapsed_ms=%s body_size=%s",
                attempt + 1,
                max(0, attempt),
                response.status_code,
                elapsed_ms,
                body_size,
            )
            response.raise_for_status()
            data = response.json()
            parsed = _parse_response(data.get("response", ""))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                return parsed["items"]
            if isinstance(parsed, dict):
                return [parsed]
            return []
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            status_code = getattr(response, "status_code", None)
            body_size = len(getattr(response, "content", b"") or b"")
            transport_audit.append(
                {
                    "attempt": attempt + 1,
                    "retry": max(0, attempt),
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                    "body_size": body_size,
                    "exception_type": type(exc).__name__,
                }
            )
            logger.warning(
                "Stage 4.1 Ollama request failed attempt=%s retry=%s status_code=%s elapsed_ms=%s body_size=%s exception=%s",
                attempt + 1,
                max(0, attempt),
                status_code,
                elapsed_ms,
                body_size,
                type(exc).__name__,
            )
            time.sleep(0.5 * (attempt + 1))
    if last_error:
        raise RelationLLMTransportError(
            f"Error invocando Ollama: {last_error}",
            audit={
                "endpoint": endpoint,
                "model": model,
                "attempts": transport_audit,
                "exception_type": type(last_error).__name__,
                "exception": str(last_error),
            },
        ) from last_error
    return []


def _build_prompt(candidates: list[dict[str, Any]], prompt_version: str) -> str:
    instructions = [
        "Responde SOLO JSON válido (array).",
        "Devuelve candidate_id EXACTAMENTE igual (literal) al recibido en input, sin cambios.",
        "No inventar ni renombrar candidate_id; usa solo IDs presentes en input.",
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
    if not str(content or "").strip():
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"_raw": content, "_parse_error": True}
        return {"_raw": content, "_parse_error": True}
