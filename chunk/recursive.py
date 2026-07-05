from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import BaseChunker
from .overlap import OverlapHandler
from .reference import ReferenceStripper
from ..core.types import Chunk


class RecursiveChunker(BaseChunker):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._overlap = OverlapHandler()
        self._ref_stripper = ReferenceStripper()
        self._strip_refs = getattr(config, "strip_references", True) if config else True
        self._overlap_sentences = getattr(config, "chunk_overlap_sentences", 2) if config else 2

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None, page_spans: list = None) -> List[Chunk]:
        if self._strip_refs:
            text = self._ref_stripper.strip(text)

        text = self._pre_clean(text)

        sections = self._split_at_headings(text)
        if not sections:
            sections = [(text, "")]

        all_chunks: List[Chunk] = []
        chunk_index = 0

        for section_text, heading in sections:
            section_chunks = self._chunk_section(section_text, heading, doc_id, chunk_index)
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        if page_spans:
            self._assign_pages(all_chunks, text, page_spans)

        if self._overlap_sentences > 0 and len(all_chunks) > 1:
            all_chunks = self._overlap.apply(all_chunks, self._overlap_sentences)

        return all_chunks

    def _assign_pages(self, chunks: list, full_text: str, page_spans: list) -> None:
        import re
        def norm(t):
            return re.sub(r"\s+", " ", t).strip()

        norm_text = norm(full_text)
        norm_to_orig = {}
        orig_pos = 0
        norm_pos = 0
        for ch in full_text:
            if ch in " \t\n\r":
                if norm_pos > 0 and norm_text[norm_pos-1:norm_pos] == " ":
                    pass
                else:
                    orig_pos += 1
                    continue
            if norm_pos < len(norm_text):
                norm_to_orig[norm_pos] = orig_pos
            norm_pos += 1
            orig_pos += 1

        for chunk in chunks:
            best_page = 0
            best_score = 0
            chunk_words = chunk.text.split()
            for window_size in [25, 20, 15, 10, 8, 6, 4]:
                if len(chunk_words) < window_size:
                    continue
                for start_idx in range(min(8, len(chunk_words) - window_size + 1)):
                    fragment = " ".join(chunk_words[start_idx:start_idx + window_size])
                    npos = norm_text.find(fragment)
                    if npos == -1:
                        continue
                    orig_char = norm_to_orig.get(npos, npos)
                    for span_start, span_end, page_num in page_spans:
                        if span_start <= orig_char < span_end:
                            if window_size > best_score:
                                best_score = window_size
                                best_page = page_num
                            break
                    if best_score >= 10:
                        break
                if best_score >= 10:
                    break

            if best_page == 0 and len(chunks) > 1:
                idx = chunk.index
                total = len(chunks)
                for span_start, span_end, page_num in page_spans:
                    frac = span_start / max(len(full_text), 1)
                    chunk_frac = idx / max(total, 1)
                    if abs(frac - chunk_frac) < 0.15:
                        best_page = page_num
                        break

            chunk.page = best_page

    def _pre_clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _split_at_headings(self, text: str) -> List[tuple]:
        pattern = re.compile(
            r"^(#{1,6})\s+(.+)$|"
            r"^\s*(?:\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{3,60})$|"
            r"^\s*(?:Chapter|Section|Part|Lesson)\s+\d+[.:]\s*(.+)$|"
            r"^\s*([A-Z][A-Z\s]{3,60})\s*$",
            re.MULTILINE,
        )

        matches = list(pattern.finditer(text))
        if not matches:
            return [(text, "")]

        sections: List[tuple] = []

        if matches[0].start() > 10:
            preamble = text[:matches[0].start()].strip()
            if preamble and self._word_count(preamble) >= self._min_words:
                sections.append((preamble, ""))

        for i, match in enumerate(matches):
            heading = ""
            for g in match.groups():
                if g and g.strip():
                    heading = g.strip()
                    break

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            if heading and body.startswith(heading):
                body = body[len(heading):].strip()
                body = re.sub(r"^[\n\s]+", "", body)

            if body:
                sections.append((body, heading))

        return sections

    def _chunk_section(
        self, text: str, heading: str, doc_id: str, start_index: int
    ) -> List[Chunk]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current_text = ""
        current_words = 0
        idx = start_index

        def flush() -> None:
            nonlocal current_text, current_words, idx
            if current_words >= self._min_words:
                final_text = current_text.strip()
                if final_text:
                    chunks.append(self._make_chunk(
                        final_text, f"chunk_{idx}", doc_id, heading, idx
                    ))
                    idx += 1
            current_text = ""
            current_words = 0

        for para in paragraphs:
            clean = re.sub(r"\s+", " ", para).strip()
            if not clean:
                continue
            para_wc = self._word_count(clean)
            if para_wc < 3:
                continue
            if current_words + para_wc > self._max_words:
                flush()
            if para_wc > self._max_words:
                sentences = self._split_sentences(clean)
                for sent in sentences:
                    sent_wc = self._word_count(sent)
                    if current_words + sent_wc > self._max_words:
                        flush()
                    if sent_wc > self._max_words:
                        words_list = sent.split()
                        for j in range(0, len(words_list), self._max_words - 10):
                            segment = " ".join(words_list[j:j + self._max_words - 10])
                            seg_wc = self._word_count(segment)
                            if current_words + seg_wc > self._max_words:
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
            else:
                if current_text:
                    current_text += " " + clean
                else:
                    current_text = clean
                current_words += para_wc

        flush()
        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]