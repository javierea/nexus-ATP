"""HTTP client interface placeholder.

This module is intentionally minimal in Etapa 0. The goal is to define
expected interfaces without implementing network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RequestConfig:
    """Configuration for future HTTP requests."""

    user_agent: str
    timeout_sec: int


class HttpClient(Protocol):
    """Protocol for an HTTP client implementation."""

    def get(self, url: str, config: RequestConfig) -> bytes:
        """Fetch a URL and return raw bytes."""
        raise NotImplementedError
