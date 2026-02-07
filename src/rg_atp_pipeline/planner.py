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

    Formats:
    - res-{year}-{n}-20-1.pdf (n without zero-padding).
    - rg-{year}-{n}-20-1.pdf (n without zero-padding).
    """
    if start_n > end_n:
        return []
    base = base_url.rstrip("/")
    urls: list[str] = []
    for n in range(start_n, end_n + 1):
        urls.append(f"{base}/res-{year}-{n}-20-1.pdf")
        urls.append(f"{base}/rg-{year}-{n}-20-1.pdf")
    return urls


def plan_old(
    base_url: str,
    start: int,
    end: int,
    min_number: int,
    year_start: int,
    year_end: int,
) -> List[str]:
    """Plan URLs for old PDFs.

    Formats:
    - {num}.pdf
    - {num}-{YYYY}.pdf
    - {num}-{YY}.pdf (YY zero-padded two-digit suffix)
    """
    if start > end:
        return []
    base = base_url.rstrip("/")
    lower_bound = max(start, min_number)
    if lower_bound > end:
        return []
    urls: list[str] = []
    for num in range(end, lower_bound - 1, -1):
        urls.append(f"{base}/{num}.pdf")
        for year in range(year_end, year_start - 1, -1):
            urls.append(f"{base}/{num}-{year}.pdf")
            urls.append(f"{base}/{num}-{year % 100:02d}.pdf")
    return urls


def plan_new_docs(base_url: str, year: int, start_n: int, end_n: int) -> List[PlannedDoc]:
    """Plan URLs for new PDFs with metadata."""
    if start_n > end_n:
        return []
    base = base_url.rstrip("/")
    docs = []
    for n in range(start_n, end_n + 1):
        res_key = f"res-{year}-{n}-20-1"
        res_url = f"{base}/{res_key}.pdf"
        docs.append(PlannedDoc(doc_key=res_key, url=res_url, doc_family="NEW", year=year, number=n))
        rg_key = f"rg-{year}-{n}-20-1"
        rg_url = f"{base}/{rg_key}.pdf"
        docs.append(PlannedDoc(doc_key=rg_key, url=rg_url, doc_family="NEW", year=year, number=n))
    return docs


def plan_old_docs(
    base_url: str,
    start: int,
    end: int,
    min_number: int,
    year_start: int,
    year_end: int,
) -> List[PlannedDoc]:
    """Plan URLs for old PDFs with metadata."""
    if start > end:
        return []
    base = base_url.rstrip("/")
    docs = []
    lower_bound = max(start, min_number)
    if lower_bound > end:
        return []
    for num in range(end, lower_bound - 1, -1):
        doc_key = f"OLD-{num}"
        url = f"{base}/{num}.pdf"
        docs.append(PlannedDoc(doc_key=doc_key, url=url, doc_family="OLD", year=None, number=num))
        for year in range(year_end, year_start - 1, -1):
            for suffix in (str(year), f"{year % 100:02d}"):
                doc_key = f"OLD-{num}-{suffix}"
                url = f"{base}/{num}-{suffix}.pdf"
                docs.append(
                    PlannedDoc(doc_key=doc_key, url=url, doc_family="OLD", year=None, number=num)
                )
    return docs


def plan_all_docs(config: Config) -> List[PlannedDoc]:
    """Plan all candidate URLs for new and old ranges with metadata."""
    docs: List[PlannedDoc] = []
    for year in config.years:
        max_n = config.max_n_by_year.get(str(year), 0)
        docs.extend(plan_new_docs(config.base_url_new, year, 1, max_n))
    docs.extend(
        plan_old_docs(
            config.base_url_old,
            config.old_range.start,
            config.old_range.end,
            config.old_min_number,
            config.old_year_start,
            config.old_year_end,
        )
    )
    return docs


def plan_all(config: Config) -> List[str]:
    """Plan all candidate URLs for new and old ranges."""
    urls: List[str] = []
    for year in config.years:
        max_n = config.max_n_by_year.get(str(year), 0)
        urls.extend(plan_new(config.base_url_new, year, 1, max_n))
    urls.extend(
        plan_old(
            config.base_url_old,
            config.old_range.start,
            config.old_range.end,
            config.old_min_number,
            config.old_year_start,
            config.old_year_end,
        )
    )
    return urls
