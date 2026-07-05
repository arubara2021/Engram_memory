from __future__ import annotations

import re
from typing import Optional


SMART_QUOTE_MAP = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
}

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00c6": "AE",
    "\u00e6": "ae",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00df": "ss",
}

HYPHENATION_RE = re.compile(r"(\w)-\n(\w)")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class TextCleaner:

    def __init__(self) -> None:
        pass

    def clean(self, text: Optional[str]) -> str:
        if not text:
            return ""

        text = self._fix_ligatures(text)
        text = self._fix_smart_quotes(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)
        text = self._remove_control_chars(text)
        text = self._collapse_blank_lines(text)
        text = text.strip()

        return text

    def _fix_ligatures(self, text: str) -> str:
        for lig, rep in LIGATURE_MAP.items():
            text = text.replace(lig, rep)
        return text

    def _fix_smart_quotes(self, text: str) -> str:
        for smart, standard in SMART_QUOTE_MAP.items():
            text = text.replace(smart, standard)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        text = HYPHENATION_RE.sub(r"\1\2", text)
        text = re.sub(r"(\w)\u2010\n(\w)", r"\1\2", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = text.replace("\t", " ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = MULTI_SPACE_RE.sub(" ", text)
        return text

    def _remove_control_chars(self, text: str) -> str:
        return CONTROL_CHAR_RE.sub("", text)

    def _collapse_blank_lines(self, text: str) -> str:
        return MULTI_NEWLINE_RE.sub("\n\n", text)