"""Lightweight HTTP client for rg_atp_pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Mapping, Tuple

import requests


class HttpClientError(RuntimeError):
    """Raised when the HTTP client exhausts retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class HttpClientConfig:
    """Configuration for HTTP requests."""

    rate_limit_rps: int
    timeout_sec: int
    max_attempts: int
    backoff_sec: int
    user_agent: str


class HttpClient:
    """Minimal requests-based HTTP client with retries and rate limiting."""

    def __init__(self, config: HttpClientConfig, logger: logging.Logger | None = None) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})
        self._logger = logger or logging.getLogger("rg_atp_pipeline.http")
        self._last_request_at: float | None = None

    def head(self, url: str) -> Tuple[int, Mapping[str, str]]:
        """Perform a HEAD request and return status code + headers."""
        response = self._request("HEAD", url)
        return response.status_code, response.headers

    def get_bytes(self, url: str) -> bytes:
        """Perform a GET request and return raw bytes."""
        response = self._request("GET", url)
        if response.status_code != 200:
            raise HttpClientError(
                f"GET {url} failed with status {response.status_code}",
                status_code=response.status_code,
            )
        return response.content

    def _request(self, method: str, url: str) -> requests.Response:
        for attempt in range(1, self._config.max_attempts + 1):
            self._rate_limit_sleep()
            try:
                response = self._session.request(
                    method,
                    url,
                    timeout=self._config.timeout_sec,
                    allow_redirects=True,
                )
                if response.status_code >= 500:
                    raise HttpClientError(
                        f"{method} {url} -> {response.status_code}",
                        status_code=response.status_code,
                    )
                return response
            except (requests.RequestException, HttpClientError) as exc:
                is_last = attempt == self._config.max_attempts
                self._logger.warning(
                    "HTTP error (%s %s) attempt %s/%s: %s",
                    method,
                    url,
                    attempt,
                    self._config.max_attempts,
                    exc,
                )
                if is_last:
                    status_code = exc.status_code if isinstance(exc, HttpClientError) else None
                    raise HttpClientError(str(exc), status_code=status_code) from exc
                backoff = self._config.backoff_sec * (2 ** (attempt - 1))
                time.sleep(backoff)
        raise HttpClientError(f"{method} {url} failed after retries")

    def _rate_limit_sleep(self) -> None:
        min_interval = 1 / self._config.rate_limit_rps
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()
