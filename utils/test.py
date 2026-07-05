from __future__ import annotations

import re
from typing import List, Tuple


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u0370-\u03ff\u1f00-\u1fff]", "", text)
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize(text)
    tokens = normalized.split()
    return [t for t in tokens if len(t) > 1]


def tokenize_simple(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return [t for t in text.split() if len(t) > 1]


def word_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.split())


def sentence_split(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text).strip()

    text, placeholders = _protect_special(text)

    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"\'$$])", text)

    sentences: List[str] = []
    for sent in raw:
        sent = _restore_special(sent, placeholders)
        sent = sent.strip()
        if sent:
            sentences.append(sent)

    return _merge_short(sentences)


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    if not tokens or n < 1:
        return []
    if n > len(tokens):
        return [" ".join(tokens)]
    result: List[str] = []
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i : i + n])
        result.append(gram)
    return result


def extract_ngrams_range(
    tokens: List[str], min_n: int = 2, max_n: int = 4
) -> List[Tuple[str, int]]:
    ngrams: List[Tuple[str, int]] = []
    n = len(tokens)
    for size in range(min_n, min(max_n + 1, n + 1)):
        for i in range(n - size + 1):
            gram = " ".join(tokens[i : i + size])
            ngrams.append((gram, size))
    return ngrams


def fix_ligatures(text: str) -> str:
    replacements = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl",
        "\u00c6": "AE", "\u00e6": "ae",
        "\u0152": "OE", "\u0153": "oe",
        "\u00df": "ss",
    }
    for lig, rep in replacements.items():
        text = text.replace(lig, rep)
    return text


def fix_hyphenation(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\u2010\n(\w)", r"\1\2", text)
    return text


def collapse_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _protect_special(text: str) -> tuple:
    import re as _re

    placeholders: List[str] = []
    idx = 0

    url_re = _re.compile(r"https?://\S+")
    email_re = _re.compile(r"\S+@\S+\.\S+")
    decimal_re = _re.compile(r"\d+\.\d+")
    ellipsis_re = _re.compile(r"\.{3,}")

    def replace(match: _re.Match) -> str:
        nonlocal idx
        ph = f"\x00PH{idx}\x00"
        placeholders.append(match.group(0))
        idx += 1
        return ph

    text = url_re.sub(replace, text)
    text = email_re.sub(replace, text)
    text = ellipsis_re.sub(replace, text)
    text = decimal_re.sub(replace, text)

    return text, placeholders


def _restore_special(text: str, placeholders: List[str]) -> str:
    for i, original in enumerate(placeholders):
        text = text.replace(f"\x00PH{i}\x00", original)
    return text


def _merge_short(sentences: List[str]) -> List[str]:
    if len(sentences) <= 1:
        return sentences

    merged: List[str] = []
    buffer = ""

    for sent in sentences:
        if buffer and len(buffer.split()) < 8:
            buffer += " " + sent
        elif buffer:
            merged.append(buffer)
            buffer = sent
        else:
            buffer = sent

    if buffer:
        merged.append(buffer)

    return merged