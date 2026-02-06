"""URL planner for rg_atp_pipeline.

Generates candidate URLs without performing HTTP requests.
"""

from __future__ import annotations

from typing import List

from .config import Config


def plan_new(base_url: str, year: int, start_n: int, end_n: int) -> List[str]:
    """Plan URLs for new PDFs.

    Format: RES-{year}-{n}-20-1.pdf (n without zero-padding).
    """
    if start_n > end_n:
        return []
    return [
        f"{base_url.rstrip('/')}/RES-{year}-{n}-20-1.pdf" for n in range(start_n, end_n + 1)
    ]


def plan_old(base_url: str, start: int, end: int) -> List[str]:
    """Plan URLs for old PDFs.

    Format: {num}.pdf
    """
    if start > end:
        return []
    return [f"{base_url.rstrip('/')}/{num}.pdf" for num in range(start, end + 1)]


def plan_all(config: Config) -> List[str]:
    """Plan all candidate URLs for new and old ranges."""
    urls: List[str] = []
    for year in config.years:
        max_n = config.max_n_by_year.get(str(year), 0)
        urls.extend(plan_new(config.base_url_new, year, 1, max_n))
    urls.extend(plan_old(config.base_url_old, config.old_range.start, config.old_range.end))
    return urls
