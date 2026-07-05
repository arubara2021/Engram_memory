from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..core.types import Chunk, Concept

from .hash_table import (
    MultiHashTable,
    extract_all_indexable_ngrams,
    extract_ngrams_from_tokens,
    normalize_text,
    tokenize_for_concepts,
    COMMON_WORDS,
    NOISE_TERMS,
)


class NgramIndexer:

    def build_index(
        self,
        chunks: List[Chunk],
        concepts: List[Concept],
        hash_tables: MultiHashTable,
    ) -> Dict[str, Set[str]]:
        ngram_index: Dict[str, Set[str]] = {}

        for chunk in chunks:
            ngrams = extract_all_indexable_ngrams(chunk)
            for ng, _ in ngrams:
                if ng not in ngram_index:
                    ngram_index[ng] = set()
                ngram_index[ng].add(chunk.chunk_id)

        for concept in concepts:
            for ngram in concept.ngrams:
                normalized = normalize_text(ngram)
                if normalized:
                    if normalized not in ngram_index:
                        ngram_index[normalized] = set()
                    for cid in concept.chunk_ids:
                        ngram_index[normalized].add(cid)

        return ngram_index

    def lookup(
        self,
        query_text: str,
        hash_tables: MultiHashTable,
        ngram_index: Optional[Dict[str, Set[str]]] = None,
    ) -> List[Tuple[str, int]]:
        query_tokens = tokenize_for_concepts(query_text)
        query_ngrams: Set[str] = set()

        for ng, _ in extract_ngrams_from_tokens(query_tokens, 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1:
                query_ngrams.add(t)

        if not query_ngrams:
            normalized = normalize_text(query_text)
            if normalized:
                query_ngrams = {normalized}

        chunk_hits: Dict[str, int] = {}

        for ngram in query_ngrams:
            entries = hash_tables.lookup_by_ngram(ngram)
            for entry in entries:
                cid = entry.chunk_id
                chunk_hits[cid] = chunk_hits.get(cid, 0) + 1

        results = sorted(chunk_hits.items(), key=lambda x: -x[1])
        return results

    def extract_query_ngrams(self, query_text: str) -> Set[str]:
        query_tokens = tokenize_for_concepts(query_text)
        query_ngrams: Set[str] = set()

        for ng, _ in extract_ngrams_from_tokens(query_tokens, 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1:
                query_ngrams.add(t)

        if not query_ngrams:
            normalized = normalize_text(query_text)
            if normalized:
                query_ngrams = {normalized}

        return query_ngrams