from __future__ import annotations

import re
from typing import List


REFERENCE_TITLE_RE = re.compile(
    r"^\s*(References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY|"
    r"References and Further Reading|Selected References)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

REFERENCE_LINE_RE = re.compile(
    r"^\s*[$$]?\d+[$$\.)]?\s*[A-Z][a-z]+.*?\d{4}"
)

REFERENCE_AUTHOR_RE = re.compile(
    r"^\s*[A-Z][a-z]+,?\s+[A-Z]\..*?\d{4}"
)

REFERENCE_AUTHOR2_RE = re.compile(
    r"^\s*[A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+.*?\d{4}"
)

URL_LINE_RE = re.compile(
    r"https?://|doi\.org|arxiv\.org|arXiv:|DOI:",
    re.IGNORECASE,
)


class ReferenceStripper:

    def strip(self, text: str, max_reference_pages: int = 15) -> str:
        lines = text.split("\n")
        ref_start = self._find_reference_heading(lines)

        if ref_start == -1:
            ref_start = self._scan_for_citation_block(lines, max_reference_pages)

        if ref_start != -1:
            cleaned = "\n".join(lines[:ref_start])
            return cleaned.rstrip()

        return text

    def _find_reference_heading(self, lines: List[str]) -> int:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if REFERENCE_TITLE_RE.match(stripped):
                return i
        return -1

    def _scan_for_citation_block(
        self, lines: List[str], max_pages: int
    ) -> int:
        avg_lines_per_page = 40
        scan_from = max(0, len(lines) - max_pages * avg_lines_per_page)

        consecutive = 0
        for i in range(len(lines) - 1, scan_from - 1, -1):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if (
                REFERENCE_LINE_RE.match(stripped)
                or REFERENCE_AUTHOR_RE.match(stripped)
                or REFERENCE_AUTHOR2_RE.match(stripped)
            ):
                consecutive += 1
                if consecutive >= 5:
                    return i
            else:
                consecutive = 0

        return -1

    def strip_url_heavy_lines(self, text: str) -> str:
        lines = text.split("\n")
        filtered: List[str] = []
        for line in lines:
            url_count = len(URL_LINE_RE.findall(line))
            if url_count > 0 and len(line.strip()) < 200:
                alpha = sum(1 for c in line if c.isalpha())
                total = len(line.strip())
                if total > 0 and alpha / total < 0.7:
                    continue
            filtered.append(line)
        return "\n".join(filtered)