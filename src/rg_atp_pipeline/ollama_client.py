"""Ollama client for missing download review."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Sequence

import requests

from .llm_review import LLMReviewer, MissingDownloadCandidate, MissingDownloadReview


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is not reachable."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    base_url: str = "http://localhost:11434"


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        payload = {
            "model": self._config.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
            },
        }
        if self._config.max_tokens is not None:
            payload["options"]["num_predict"] = self._config.max_tokens
        base_url = self._config.base_url.rstrip("/")
        endpoint = base_url if base_url.endswith("/api/chat") else f"{base_url}/api/chat"
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Ollama no disponible en {self._config.base_url}"
            ) from exc
        if response.status_code != 200:
            raise OllamaUnavailableError(
                f"Ollama respondió {response.status_code}: {response.text}"
            )
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "")


class OllamaReviewer(LLMReviewer):
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    def review(
        self, items: Sequence[MissingDownloadCandidate]
    ) -> list[MissingDownloadReview]:
        results: list[MissingDownloadReview] = []
        for item in items:
            prompt = _build_prompt(item)
            content = self._client.chat(prompt)
            parsed = _parse_response(content)
            results.append(
                MissingDownloadReview(
                    doc_key=parsed.get("doc_key", item.doc_key),
                    verdict=parsed.get("verdict", "UNKNOWN"),
                    org_guess=parsed.get("org_guess", "Unknown"),
                    confidence=_parse_confidence(
                        parsed.get("confidence"), item.doc_key
                    ),
                    reason=parsed.get("reason", "").strip()[:280],
                    status=item.status,
                    url=item.url,
                    last_checked_at=item.last_checked_at,
                    last_downloaded_at=item.last_downloaded_at,
                )
            )
        return results


def _build_prompt(item: MissingDownloadCandidate) -> list[dict[str, str]]:
    evidence = item.evidence_snippet
    if len(evidence) > 300:
        evidence = evidence[:300]
    user_payload = {
        "doc_key_normalized": item.doc_key,
        "raw_reference": item.raw_reference,
        "evidence_snippet": evidence,
        "page_number": item.page_number,
        "rules": [
            "Si la evidencia menciona AFIP/ARCA, Comisión Arbitral/C.A. u otro organismo ≠ ATP -> OTHER_ORG.",
            "Si el número parece venir de miles/decimales (ej 1.895 -> OLD-1) -> DETECTION_ERROR.",
            "Si menciona ATP/Administración Tributaria Provincial/Chaco o DGR (nombre anterior de ATP) -> ATP_MISSING.",
            "Si no hay señales claras -> UNKNOWN.",
        ],
        "output_schema": {
            "doc_key": "RES-2024-39-20-1 | OLD-2172 | OLD-1",
            "verdict": "ATP_MISSING | OTHER_ORG | DETECTION_ERROR | UNKNOWN",
            "org_guess": (
                "ATP Chaco | AFIP/ARCA | Comisión Arbitral/C.A. | "
                "AGIP/Otro | Unknown"
            ),
            "confidence": "0.0-1.0",
            "reason": "frase corta basada SOLO en evidencia",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Eres un clasificador de referencias normativas. "
                "No hagas interpretación jurídica. "
                "Responde SOLO con JSON válido y sin texto adicional."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _parse_response(content: str) -> dict[str, object]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _parse_confidence(value: object, doc_key: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return _default_confidence(value, doc_key)
        if "-" in cleaned:
            cleaned = cleaned.split("-", 1)[0].strip()
        try:
            return float(cleaned)
        except ValueError:
            return _default_confidence(value, doc_key)
    if value is None:
        return _default_confidence(value, doc_key)
    return _default_confidence(value, doc_key)


def _default_confidence(value: object, doc_key: str) -> float:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Ollama confidence inválido para %s: %r. Usando 0.0.", doc_key, value
    )
    return 0.0
