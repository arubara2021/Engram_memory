from __future__ import annotations

import re
from typing import List


URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
DECIMAL_RE = re.compile(r"\d+\.\d+")
ELLIPSIS_RE = re.compile(r"\.{3,}")
NUMBERED_LIST_RE = re.compile(r"^\d+[.)]\s", re.MULTILINE)
LETTERED_LIST_RE = re.compile(r"^\(?[a-z]$$\s", re.MULTILINE)
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")
SHORT_SENTENCE_WORDS = 8


class SentenceSplitter:

    def __init__(self) -> None:
        self._min_sentence_words = 3

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        text, placeholders = self._protect_special(text)

        raw_sentences = SENTENCE_BREAK_RE.split(text)

        sentences: List[str] = []
        for sent in raw_sentences:
            sent = self._restore_special(sent, placeholders)
            sent = sent.strip()
            if sent:
                sentences.append(sent)

        sentences = self._merge_short(sentences)
        return sentences

    def split_into_chunks(self, text: str, max_words: int) -> List[str]:
        sentences = self.split(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_count = 0

        for sent in sentences:
            words = sent.split()
            if current_count + len(words) > max_words and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_count = len(words)
            else:
                current.append(sent)
                current_count += len(words)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def count_sentences(self, text: str) -> int:
        return len(self.split(text))

    def _protect_special(self, text: str) -> tuple:
        placeholders: List[str] = []

        all_patterns = [URL_RE, EMAIL_RE, ELLIPSIS_RE, DECIMAL_RE]

        for pattern in all_patterns:
            for match in pattern.finditer(text):
                placeholders.append(match.group(0))

        for idx, ph_text in enumerate(placeholders):
            text = text.replace(ph_text, f"\x00PH{idx}\x00", 1)

        return text, placeholders

    def _restore_special(self, text: str, placeholders: List[str]) -> str:
        for i, original in enumerate(placeholders):
            text = text.replace(f"\x00PH{i}\x00", original)
        return text

    def _merge_short(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= 1:
            return sentences

        merged: List[str] = []
        buffer = ""

        for sent in sentences:
            if buffer and len(buffer.split()) < SHORT_SENTENCE_WORDS:
                buffer += " " + sent
            elif buffer:
                merged.append(buffer)
                buffer = sent
            else:
                buffer = sent

        if buffer:
            merged.append(buffer)

        return merged
