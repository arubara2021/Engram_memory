from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker
from ..core.types import Chunk


class PageChunker(BaseChunker):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._merge_short = True

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        pages: Optional[List[Tuple[int, str]]] = None,
    ) -> List[Chunk]:
        if pages is None:
            pages = self._split_by_page_breaks(text)

        if not pages:
            pages = [(1, text)]

        chunks: List[Chunk] = []
        idx = 0

        for page_num, page_text in pages:
            clean = re.sub(r"\s+", " ", page_text).strip()
            if not clean:
                continue
            wc = self._word_count(clean)

            if wc < self._min_words and self._merge_short and chunks:
                last = chunks[-1]
                combined = last.text + "\n\n" + clean
                if self._word_count(combined) <= self._max_words:
                    chunks[-1] = self._make_chunk(
                        combined, last.chunk_id, doc_id, "", last.index, last.page
                    )
                    continue
                elif self._word_count(combined) <= self._max_words * 2:
                    chunks[-1] = self._make_chunk(
                        combined, last.chunk_id, doc_id, "", last.index, last.page
                    )
                    continue

            if wc > self._max_words:
                sub_chunks = self._split_page(clean, page_num, doc_id, idx)
                chunks.extend(sub_chunks)
                idx += len(sub_chunks)
            else:
                chunks.append(self._make_chunk(
                    clean, f"chunk_{idx}", doc_id, "", idx, page_num
                ))
                idx += 1

        return chunks

    def _split_by_page_breaks(self, text: str) -> List[Tuple[int, str]]:
        pattern = re.compile(
            r"\f|"
            r"(?:^|\n)\s*(?:page|p\.?)\s*\d+\s*(?:of\s*\d+)?\s*(?:\n|$)|"
            r"(?:^|\n)\s*\d+\s*/\s*\d+\s*(?:\n|$)",
            re.IGNORECASE | re.MULTILINE,
        )

        parts = pattern.split(text)
        pages: List[Tuple[int, str]] = []

        for i, part in enumerate(parts):
            clean = part.strip()
            if clean:
                pages.append((i + 1, clean))

        if not pages:
            return [(1, text)]

        return pages

    def _split_page(
        self, text: str, page_num: int, doc_id: str, start_idx: int
    ) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        current_text = ""
        current_wc = 0
        idx = start_idx

        for sent in sentences:
            sent_wc = self._word_count(sent)
            if current_wc + sent_wc > self._max_words and current_text.strip():
                if current_wc >= self._min_words:
                    chunks.append(self._make_chunk(
                        current_text.strip(),
                        f"chunk_{idx}",
                        doc_id,
                        "",
                        idx,
                        page_num,
                    ))
                    idx += 1
                current_text = ""
                current_wc = 0
            if current_text:
                current_text += " " + sent
            else:
                current_text = sent
            current_wc += sent_wc

        if current_text.strip() and current_wc >= self._min_words:
            chunks.append(self._make_chunk(
                current_text.strip(),
                f"chunk_{idx}",
                doc_id,
                "",
                idx,
                page_num,
            ))

        return chunks