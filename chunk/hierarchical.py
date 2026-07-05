from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import BaseChunker
from .overlap import OverlapHandler
from .reference import ReferenceStripper
from ..core.types import Chunk


class HierarchicalChunker(BaseChunker):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._overlap = OverlapHandler()
        self._ref_stripper = ReferenceStripper()
        self._small_size = getattr(config, "hierarchical_small", 200) if config else 200
        self._large_size = getattr(config, "hierarchical_large", 800) if config else 800
        self._strip_refs = getattr(config, "strip_references", True) if config else True
        self._overlap_sentences = getattr(config, "chunk_overlap_sentences", 2) if config else 2

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if self._strip_refs:
            text = self._ref_stripper.strip(text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        small_chunks = self._create_small_chunks(text, doc_id)
        large_chunks = self._create_large_chunks(small_chunks, doc_id)

        for sc in small_chunks:
            for lc in large_chunks:
                if sc.text in lc.text:
                    sc.section = f"parent:{lc.chunk_id}"
                    break

        all_chunks = list(small_chunks) + list(large_chunks)

        if self._overlap_sentences > 0 and len(small_chunks) > 1:
            overlapped = self._overlap.apply(list(small_chunks), self._overlap_sentences)
            id_map = {c.chunk_id: c for c in overlapped}
            for i, sc in enumerate(all_chunks):
                if sc.chunk_id in id_map and sc.chunk_id.startswith("small_"):
                    all_chunks[i] = id_map[sc.chunk_id]

        return all_chunks

    def _create_small_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_text = ""
        current_words = 0
        idx = 0

        def flush() -> None:
            nonlocal current_text, current_words, idx
            if current_words >= self._min_words and current_text.strip():
                chunks.append(self._make_chunk(
                    current_text.strip(),
                    f"small_{idx}",
                    doc_id,
                    "",
                    idx,
                ))
                idx += 1
            current_text = ""
            current_words = 0

        for sent in sentences:
            sent_wc = self._word_count(sent)
            if current_words + sent_wc > self._small_size:
                flush()
            if sent_wc > self._small_size:
                words_list = sent.split()
                for j in range(0, len(words_list), self._small_size - 10):
                    segment = " ".join(words_list[j:j + self._small_size - 10])
                    seg_wc = self._word_count(segment)
                    if current_words + seg_wc > self._small_size:
                        flush()
                    if current_text:
                        current_text += " " + segment
                    else:
                        current_text = segment
                    current_words += seg_wc
            else:
                if current_text:
                    current_text += " " + sent
                else:
                    current_text = sent
                current_words += sent_wc

        flush()
        return chunks

    def _create_large_chunks(self, small_chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        if not small_chunks:
            return []

        large_chunks: List[Chunk] = []
        group: List[Chunk] = []
        group_words = 0
        idx = 0

        for sc in small_chunks:
            if group_words + sc.word_count > self._large_size and group:
                merged_text = " ".join(c.text for c in group)
                large_chunks.append(self._make_chunk(
                    merged_text,
                    f"large_{idx}",
                    doc_id,
                    "",
                    idx,
                ))
                idx += 1
                group = []
                group_words = 0
            group.append(sc)
            group_words += sc.word_count

        if group:
            merged_text = " ".join(c.text for c in group)
            large_chunks.append(self._make_chunk(
                merged_text,
                f"large_{idx}",
                doc_id,
                "",
                idx,
            ))

        return large_chunks