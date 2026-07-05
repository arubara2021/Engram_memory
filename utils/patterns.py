from __future__ import annotations

import re
from typing import List


DEFINITION_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"([A-Z][\w\s-]{2,30}?)\s+(?:is defined as|is a type of|is a method|"
        r"is a technique|is a process|is the process|is the ability|is the sum|"
        r"is the product|is the result|is the number|is the set|is the ratio|"
        r"is the average|is a function|is a measure|is a metric)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][\w\s-]{2,30}?)\s+(?:refers to|denotes|represents|describes|"
        r"computes|calculates|determines|measures|quantifies|estimates|"
        r"predicts|classifies|generates|produces|transforms|maps|converts|"
        r"encodes|decodes)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the\s+)?([A-Z][\w\s-]{2,30}?)\s+(?:consists of|involves|"
        r"includes|comprises|contains|requires|uses|utilizes|employs|"
        r"leverages|exploits|applies|implements|executes|performs)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:we define|let us define|define)\s+"
        r"([A-Z][\w\s-]{2,30}?)(?:\s+as|\s+to be)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][\w\s-]{2,30}?)\s*$$([A-Z]{2,6})$$\s+(?:is|refers|denotes)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:known as|called|termed|referred to as)\s+([A-Z][\w\s-]{2,30}?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Definition|Theorem|Lemma|Proposition|Property|Corollary)"
        r"\s*[:\d.]*\s*[:.]?\s*([A-Z][\w\s-]{2,30}?)\s+(?:is|are|denotes|refers)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([a-z][\w-]{3,25}?)\s+(?:is defined as|is a type of|refers to|is the process)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([a-z][\w-]{3,25}?)\s+(?:is a|are a)\s+(?:method|technique|approach|"
        r"mechanism|process|function|operation|layer|network|model|algorithm|"
        r"architecture|component|module|unit|block|structure|pattern)",
        re.IGNORECASE,
    ),
]

HEADING_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(?:\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{3,60})", re.MULTILINE),
    re.compile(
        r"^\s*(?:Chapter|Section|Part|Lesson)\s+\d+[.:]\s*([A-Z][^\n]{3,60})",
        re.MULTILINE,
    ),
    re.compile(r"^\s*([A-Z][A-Z\s]{3,60})\s*$", re.MULTILINE),
    re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),
]

REFERENCE_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"^\s*(References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*[$$]?\d+[$$\.)]?\s*[A-Z][a-z]+.*?\d{4}", re.MULTILINE
    ),
    re.compile(
        r"^\s*[A-Z][a-z]+,?\s+[A-Z]\..*?\d{4}", re.MULTILINE
    ),
    re.compile(
        r"^\s*[A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+.*?\d{4}", re.MULTILINE
    ),
]

CITATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"$$\d+(?:\s*,\s*\d+)*(?:\s*-\s*\d+)?$$"),
    re.compile(
        r"$$\s*[A-Z][a-z]+(?:\s+(?:et al|and|&)\s+[A-Z][a-z]+)?"
        r",?\s+\d{4}\s*$$"
    ),
    re.compile(r"(?:doi|DOI):\s*\S+"),
    re.compile(r"arXiv:\d{4}\.\d{4,5}"),
    re.compile(r"$$\d+$$"),
]

TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^#\s+(.+)$", re.MULTILINE),
    re.compile(
        r"^\s*(?:Chapter|Section|Part)\s+1[:\s]+(.+)$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*([A-Z][A-Z\s]{5,60})\s*$", re.MULTILINE),
    re.compile(r"^\s*([A-Z][^\n]{5,80})\s*$", re.MULTILINE),
]

PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"\'\(])")
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPECIAL_CHAR_RE = re.compile(r"[^\w\s]")
CAMEL_CASE_RE = re.compile(r"([a-z])([A-Z])")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")