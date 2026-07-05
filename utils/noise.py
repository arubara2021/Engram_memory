from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .common_words import COMMON_WORDS, NOISE_TERMS


URL_PATTERN = re.compile(
    r"https?://|www\.|\.org/|\.com/|\.edu/|\.net/|doi\.org|arxiv\.org|"
    r"arXiv:|DOI:|ISBN|ISSN|pp\.\s*\d|vol\.\s*\d|"
    r"\d{4}\.\d{4,5}|arXiv\s+\d{4}\.\d{4,5}",
    re.IGNORECASE,
)

REFERENCE_TITLE_RE = re.compile(
    r"^\s*(References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY|"
    r"References and Further Reading|Selected References)\s*",
    re.IGNORECASE | re.MULTILINE,
)

REFERENCE_LINE_RE = re.compile(
    r"^\s*[\[]?\d+[\]\.)]?\s*[A-Z][a-z]+.*?\d{4}"
)

REFERENCE_AUTHOR_RE = re.compile(
    r"^\s*[A-Z][a-z]+,?\s+[A-Z]\..*?\d{4}"
)

REFERENCE_AUTHOR2_RE = re.compile(
    r"^\s*[A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+.*?\d{4}"
)

CITATION_RE = re.compile(
    r"\[[\d,\s-]+\]|\(\s*[A-Z][a-z]+(?:\s+(?:et al|and|&)\s+[A-Z][a-z]+)?,?\s+\d{4}\s*\)$$"
)


def is_url_noise(text: str) -> bool:
    if URL_PATTERN.search(text):
        return True
    words = text.lower().split()
    if not words:
        return False
    noise_count = sum(1 for w in words if w in NOISE_TERMS)
    if len(words) > 0 and noise_count / len(words) > 0.4:
        return True
    return False


def is_reference_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if REFERENCE_TITLE_RE.match(stripped):
        return True
    if REFERENCE_LINE_RE.match(stripped):
        return True
    if REFERENCE_AUTHOR_RE.match(stripped):
        return True
    if REFERENCE_AUTHOR2_RE.match(stripped):
        return True
    return False


def is_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def contains_noise_terms(text: str) -> bool:
    words = text.lower().split()
    if not words:
        return False
    noise_count = sum(1 for w in words if w in NOISE_TERMS)
    return noise_count / len(words) > 0.4


def filter_concept_noise(concept: Dict[str, Any]) -> bool:
    term = concept.get("term", "")
    if not term:
        return False
    if is_url_noise(term):
        return False
    if term.strip().lower() in NOISE_TERMS:
        return False
    words = term.lower().split()
    if all(w in NOISE_TERMS or w in COMMON_WORDS for w in words):
        return False
    if any(c.isdigit() for c in term) and len(term.split()) <= 2:
        if not any(c.isalpha() for c in term):
            return False
    if len(term.strip()) < 2:
        return False
    return True


def filter_url_lines(text: str) -> str:
    lines = text.split("\n")
    filtered: List[str] = []
    for line in lines:
        url_count = len(URL_PATTERN.findall(line))
        if url_count > 0 and len(line.strip()) < 200:
            alpha = sum(1 for c in line if c.isalpha())
            total = len(line.strip())
            if total > 0 and alpha / total < 0.7:
                continue
        filtered.append(line)
    return "\n".join(filtered)


def filter_reference_section(text: str) -> str:
    lines = text.split("\n")
    ref_start = -1

    for i, line in enumerate(lines):
        if REFERENCE_TITLE_RE.match(line.strip()):
            ref_start = i
            break

    if ref_start == -1:
        consecutive = 0
        for i in range(len(lines) - 1, max(0, len(lines) - 300), -1):
            stripped = lines[i].strip()
            if (
                REFERENCE_LINE_RE.match(stripped)
                or REFERENCE_AUTHOR_RE.match(stripped)
                or REFERENCE_AUTHOR2_RE.match(stripped)
            ):
                consecutive += 1
                if consecutive >= 5:
                    ref_start = i
                    break
            else:
                consecutive = 0

    if ref_start != -1:
        return "\n".join(lines[:ref_start])
    return text