from __future__ import annotations

import re
from typing import List

from ..core.types import Chunk


class OverlapHandler:

    def apply(self, chunks: List[Chunk], overlap_sentences: int = 2) -> List[Chunk]:
        if len(chunks) <= 1 or overlap_sentences <= 0:
            return chunks

        result: List[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            tail = self._get_tail_sentences(prev.text, overlap_sentences)
            if tail:
                new_text = tail + " " + curr.text
                new_words = curr.words[:]
                tail_words = self._tokenize_simple(tail)
                merged_words = tail_words + new_words

                updated = Chunk(
                    chunk_id=curr.chunk_id,
                    doc_id=curr.doc_id,
                    text=new_text.strip(),
                    word_count=len(merged_words),
                    page=curr.page,
                    section=curr.section,
                    index=curr.index,
                    embedding=curr.embedding,
                    concepts=curr.concepts,
                    words=merged_words,
                    original_text=curr.original_text,
                )
                result.append(updated)
            else:
                result.append(curr)

        return result

    def _get_tail_sentences(self, text: str, n: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= n:
            return text.strip()
        tail = sentences[-n:]
        return " ".join(tail)

    def _tokenize_simple(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return [t for t in text.split() if len(t) > 1]