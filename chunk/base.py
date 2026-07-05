from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.types import Chunk


class BaseChunker(ABC):

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._min_words = getattr(config, "min_chunk_words", 20) if config else 20
        self._max_words = getattr(config, "max_chunk_words", 180) if config else 180

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _tokenize(self, text: str) -> List[str]:
        normalized = text.lower().strip()
        normalized = re.sub(r"[\u0370-\u03ff\u1f00-\u1fff]", "", normalized)
        normalized = re.sub(r"[-_]", " ", normalized)
        normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        tokens = normalized.split()
        return [t for t in tokens if len(t) > 1]

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _make_chunk(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
        section: str,
        index: int,
        page: int = 0,
    ) -> Chunk:
        words = self._tokenize(text)
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text.strip(),
            word_count=len(words),
            page=page,
            section=section,
            index=index,
            words=words,
            original_text=text,
        )

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
        if len(raw) <= 1:
            raw = re.split(r"(?<=[.!?;])\s+", text)
        if len(raw) <= 1:
            raw = re.split(r"\n\s*\n", text)
        sentences = []
        for s in raw:
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences