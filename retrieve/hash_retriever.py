from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from ..core.types import Chunk, ChunkResult

from ..index.hash_table import (
    COMMON_WORDS,
    NOISE_TERMS,
    MultiHashTable,
    extract_ngrams_from_tokens,
    normalize_text,
    tokenize_for_concepts,
)


SUFFIXES = [
    "ting", "ing", "tion", "sion", "ment", "ness", "able", "ible",
    "ful", "ous", "ive", "ory", "ize", "ise", "ify", "ate",
    "ent", "ant", "est", "ers", "ies", "led", "red",
    "ed", "er", "es", "ly", "al", "th", "ty",
]

TOC_PATTERNS = [
    re.compile(r"\.{3,}"),
    re.compile(r"\.{2,}\s*\d+"),
    re.compile(r"\d+\s*$"),
    re.compile(r"^[A-Z][A-Z\s&]{5,}$"),
]


def stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def is_toc_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    for pattern in TOC_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def content_word_overlap(
    query_tokens: List[str], chunk_words: List[str]
) -> float:
    query_stems = {
        stem(w) for w in query_tokens
        if w not in COMMON_WORDS and w not in NOISE_TERMS and len(w) > 2
    }
    chunk_stems = {
        stem(w) for w in chunk_words
        if w not in COMMON_WORDS and w not in NOISE_TERMS and len(w) > 2
    }
    if not query_stems:
        return 0.0
    return len(query_stems & chunk_stems) / len(query_stems)


def adaptive_min_overlap(query_content_words: List[str]) -> float:
    n = len(query_content_words)
    if n <= 1:
        return 0.0
    if n == 2:
        return 0.30
    if n == 3:
        return 0.25
    if n <= 5:
        return 0.20
    return 0.15


class HashRetriever:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._fp_penalty = (
            getattr(config, "false_positive_penalty", 0.3) if config else 0.3
        )
        self._top_k = (
            getattr(config, "rerank_top_k", 20) if config else 20
        )

    def retrieve(
        self,
        query: str,
        hash_tables: MultiHashTable,
        chunks: List[Chunk],
        top_k: int = 20,
        doc_id: Optional[str] = None,
    ) -> List[ChunkResult]:
        if not hash_tables or not chunks:
            return []

        if doc_id:
            chunks = [c for c in chunks if c.doc_id == doc_id]
            if not chunks:
                return []

        query_tokens = tokenize_for_concepts(query)
        query_ngrams: Set[str] = set()
        query_content_words: List[str] = []

        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1:
                query_content_words.append(t)

        for ng, _ in extract_ngrams_from_tokens(query_tokens, 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1:
                query_ngrams.add(t)
                s = stem(t)
                if s != t:
                    query_ngrams.add(s)

        if not query_ngrams:
            normalized = normalize_text(query)
            if normalized:
                query_ngrams = {normalized}

        if not query_ngrams:
            return []

        chunk_matches = self._hash_lookup(
            query_ngrams, hash_tables, chunks, doc_id
        )

        if not chunk_matches:
            chunk_matches = self._brute_force(
                query_tokens, chunks, query_content_words
            )

        chunk_map = {c.chunk_id: c for c in chunks}
        min_overlap = adaptive_min_overlap(query_content_words)
        query_len = len(query_tokens)

        results: List[ChunkResult] = []
        for cid, data in chunk_matches.items():
            chunk = chunk_map.get(cid)
            if not chunk:
                continue

            overlap = content_word_overlap(query_content_words, chunk.words)

            if data["hit_count"] == 0 and overlap < min_overlap:
                continue

            toc_ratio = self._toc_ratio(chunk.text)
            toc_penalty = toc_ratio * 0.4

            hit_count = data["hit_count"]
            matched_ngrams = data["matched_ngrams"]

            multi_word_hits = sum(1 for ng in matched_ngrams if " " in ng)
            single_word_hits = sum(
                1 for ng in matched_ngrams if " " not in ng
            )

            total_q = len(query_ngrams)
            ngram_ratio = hit_count / max(total_q, 1)

            if multi_word_hits > 0 and single_word_hits > 0:
                score = (
                    ngram_ratio * 0.35
                    + overlap * 0.30
                    + (multi_word_hits / max(total_q, 1)) * 0.20
                    + 0.15
                )
            elif multi_word_hits > 0:
                score = (
                    ngram_ratio * 0.40
                    + overlap * 0.35
                    + (multi_word_hits / max(total_q, 1)) * 0.25
                )
            else:
                score = ngram_ratio * 0.50 + overlap * 0.50

            if query_len > 1 and hit_count == 1 and multi_word_hits == 0:
                score *= 0.4

            score -= toc_penalty
            score = max(score, 0.0)

            size_bonus = data["max_ngram_size"] * 0.02
            final_score = min(score + size_bonus, 1.0)

            results.append(
                ChunkResult(
                    chunk=chunk,
                    score=final_score,
                    source="hash",
                    hash_hits=hit_count,
                    vector_sim=0.0,
                    rerank_score=0.0,
                    match_details={
                        "hit_count": hit_count,
                        "multi_word_hits": multi_word_hits,
                        "single_word_hits": single_word_hits,
                        "content_overlap": round(overlap, 3),
                        "toc_penalty": round(toc_penalty, 3),
                    },
                )
            )

        results.sort(key=lambda r: (-r.score, -r.hash_hits, r.chunk.index))
        return results[:top_k]

    def _hash_lookup(
        self,
        query_ngrams: Set[str],
        hash_tables: MultiHashTable,
        chunks: List[Chunk],
        doc_id: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        valid_ids = {c.chunk_id for c in chunks}
        chunk_matches: Dict[str, Dict[str, Any]] = {}

        for ngram_text in query_ngrams:
            indices = hash_tables.lookup_by_ngram(ngram_text)
            if not indices:
                from ..index.hash_table import hash_all

                raw = hash_all(ngram_text, hash_tables.table_size)
                for table_idx, slot_idx in enumerate(raw):
                    entries = hash_tables.tables[table_idx].get(slot_idx, [])
                    for entry in entries:
                        if entry.ngram_text == ngram_text:
                            indices.append(entry)

            for entry in indices:
                if entry.chunk_id not in valid_ids:
                    continue
                cid = entry.chunk_id
                if cid not in chunk_matches:
                    chunk_matches[cid] = {
                        "hit_count": 0,
                        "matched_ngrams": set(),
                        "max_ngram_size": 0,
                        "entries": [],
                    }
                cm = chunk_matches[cid]
                if ngram_text not in cm["matched_ngrams"]:
                    cm["hit_count"] += 1
                    cm["matched_ngrams"].add(ngram_text)
                    cm["max_ngram_size"] = max(
                        cm["max_ngram_size"], entry.ngram_size
                    )
                    cm["entries"].append(entry)

        return chunk_matches

    def _brute_force(
        self,
        query_tokens: List[str],
        chunks: List[Chunk],
        query_content_words: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        min_overlap = adaptive_min_overlap(query_content_words)
        chunk_matches: Dict[str, Dict[str, Any]] = {}

        query_stems: Set[str] = set()
        for t in query_content_words:
            query_stems.add(t)
            query_stems.add(stem(t))

        for chunk in chunks:
            chunk_stems: Set[str] = set()
            for w in chunk.words:
                chunk_stems.add(w)
                chunk_stems.add(stem(w))

            matching = query_stems & chunk_stems
            if not matching:
                continue

            overlap = content_word_overlap(query_content_words, chunk.words)
            if overlap < min_overlap:
                continue

            chunk_matches[chunk.chunk_id] = {
                "hit_count": len(matching),
                "matched_ngrams": matching,
                "max_ngram_size": 2,
                "entries": [],
            }

        return chunk_matches

    def _toc_ratio(self, text: str) -> float:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return 0.0
        toc_count = sum(1 for l in lines if is_toc_line(l))
        return toc_count / len(lines)
