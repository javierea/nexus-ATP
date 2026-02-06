"""URL planner for rg_atp_pipeline.

Generates candidate URLs without performing HTTP requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Config


@dataclass(frozen=True)
class PlannedDoc:
    """Planned document entry."""

    doc_key: str
    url: str
    doc_family: str
    year: int | None
    number: int


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


def plan_new_docs(base_url: str, year: int, start_n: int, end_n: int) -> List[PlannedDoc]:
    """Plan URLs for new PDFs with metadata."""
    if start_n > end_n:
        return []
    base = base_url.rstrip("/")
    docs = []
    for n in range(start_n, end_n + 1):
        doc_key = f"RES-{year}-{n}-20-1"
        url = f"{base}/{doc_key}.pdf"
        docs.append(PlannedDoc(doc_key=doc_key, url=url, doc_family="NEW", year=year, number=n))
    return docs


def plan_old_docs(base_url: str, start: int, end: int) -> List[PlannedDoc]:
    """Plan URLs for old PDFs with metadata."""
    if start > end:
        return []
    base = base_url.rstrip("/")
    docs = []
    for num in range(start, end + 1):
        doc_key = f"OLD-{num}"
        url = f"{base}/{num}.pdf"
        docs.append(PlannedDoc(doc_key=doc_key, url=url, doc_family="OLD", year=None, number=num))
    return docs


def plan_all_docs(config: Config) -> List[PlannedDoc]:
    """Plan all candidate URLs for new and old ranges with metadata."""
    docs: List[PlannedDoc] = []
    for year in config.years:
        max_n = config.max_n_by_year.get(str(year), 0)
        docs.extend(plan_new_docs(config.base_url_new, year, 1, max_n))
    docs.extend(plan_old_docs(config.base_url_old, config.old_range.start, config.old_range.end))
    return docs


def plan_all(config: Config) -> List[str]:
    """Plan all candidate URLs for new and old ranges."""
    urls: List[str] = []
    for year in config.years:
        max_n = config.max_n_by_year.get(str(year), 0)
        urls.extend(plan_new(config.base_url_new, year, 1, max_n))
    urls.extend(plan_old(config.base_url_old, config.old_range.start, config.old_range.end))
    return urls
