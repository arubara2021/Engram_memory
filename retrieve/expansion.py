from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..core.types import Chunk, ChunkResult

from ..index.hash_table import (
    COMMON_WORDS,
    NOISE_TERMS,
    MultiHashTable,
    extract_ngrams_from_tokens,
    tokenize_for_concepts,
)


class QueryExpander:

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def expand(
        self,
        query: str,
        results: List[ChunkResult],
        hash_tables: MultiHashTable,
        chunks: List[Chunk],
        concepts: Optional[List[Dict]] = None,
        concept_graph: Any = None,
        target: int = 3,
    ) -> List[ChunkResult]:
        if len(results) >= target:
            return results

        all_results = list(results)
        seen = {r.chunk.chunk_id for r in all_results}
        chunk_map = {c.chunk_id: c for c in chunks}

        expanded = self._expand_unigrams(query, hash_tables, chunk_map, seen)
        for r in expanded:
            r.score *= 0.7
            all_results.append(r)
            seen.add(r.chunk.chunk_id)
        if len(all_results) >= target:
            return all_results[:target]

        if concepts:
            expanded = self._expand_concept_terms(
                query, concepts, hash_tables, chunk_map, seen
            )
            for r in expanded:
                r.score *= 0.5
                all_results.append(r)
                seen.add(r.chunk.chunk_id)
            if len(all_results) >= target:
                return all_results[:target]

        if concept_graph and concepts:
            expanded = self._expand_graph(
                query, concepts, concept_graph, hash_tables, chunk_map, seen
            )
            for r in expanded:
                r.score *= 0.3
                all_results.append(r)
                seen.add(r.chunk.chunk_id)

        all_results.sort(key=lambda r: -r.score)
        return all_results[:target]

    def _expand_unigrams(
        self,
        query: str,
        hash_tables: MultiHashTable,
        chunk_map: Dict[str, Chunk],
        seen: Set[str],
    ) -> List[ChunkResult]:
        tokens = tokenize_for_concepts(query)
        results: List[ChunkResult] = []

        for token in tokens:
            if token in COMMON_WORDS or token in NOISE_TERMS or len(token) <= 2:
                continue

            entries = hash_tables.lookup_by_ngram(token)
            for entry in entries:
                if entry.chunk_id not in seen:
                    chunk = chunk_map.get(entry.chunk_id)
                    if chunk:
                        results.append(
                            ChunkResult(
                                chunk=chunk,
                                score=entry.relevance_score,
                                source="expansion",
                                hash_hits=1,
                                match_details={"expanded_with": token},
                            )
                        )
                        seen.add(entry.chunk_id)

        results.sort(key=lambda r: -r.score)
        return results

    def _expand_concept_terms(
        self,
        query: str,
        concepts: List[Dict],
        hash_tables: MultiHashTable,
        chunk_map: Dict[str, Chunk],
        seen: Set[str],
    ) -> List[ChunkResult]:
        query_lower = query.lower()
        results: List[ChunkResult] = []

        related_concepts: List[Dict] = []
        for c in concepts:
            term = c.get("term", "")
            term_words = set(term.lower().split())
            query_words = set(query_lower.split())
            overlap = len(term_words & query_words)
            if overlap > 0 and term.lower() != query_lower:
                related_concepts.append((overlap, c))

        related_concepts.sort(key=lambda x: -x[0])

        for _, concept in related_concepts[:5]:
            definition = concept.get("definition", "")
            if not definition:
                continue

            key_phrases = tokenize_for_concepts(definition[:300])
            expanded_query = " ".join(key_phrases[:10])
            if not expanded_query.strip():
                continue

            entries = hash_tables.lookup_by_ngram(expanded_query)
            for entry in entries:
                if entry.chunk_id not in seen:
                    chunk = chunk_map.get(entry.chunk_id)
                    if chunk:
                        results.append(
                            ChunkResult(
                                chunk=chunk,
                                score=entry.relevance_score,
                                source="expansion",
                                hash_hits=1,
                                match_details={
                                    "expanded_from_concept": concept.get("term", "")
                                },
                            )
                        )
                        seen.add(entry.chunk_id)

        results.sort(key=lambda r: -r.score)
        return results

    def _expand_graph(
        self,
        query: str,
        concepts: List[Dict],
        concept_graph: Any,
        hash_tables: MultiHashTable,
        chunk_map: Dict[str, Chunk],
        seen: Set[str],
    ) -> List[ChunkResult]:
        query_lower = query.lower()
        results: List[ChunkResult] = []

        matching_concepts: List[str] = []
        for c in concepts:
            term = c.get("term", "")
            if term.lower() in query_lower or query_lower in term.lower():
                matching_concepts.append(term)

        for mc in matching_concepts[:3]:
            related = concept_graph.get_related(mc, max_depth=1)
            for rel_term in related:
                entries = hash_tables.lookup(rel_term)
                for entry in entries:
                    if entry.chunk_id not in seen:
                        chunk = chunk_map.get(entry.chunk_id)
                        if chunk:
                            results.append(
                                ChunkResult(
                                    chunk=chunk,
                                    score=entry.relevance_score,
                                    source="expansion",
                                    hash_hits=1,
                                    match_details={
                                        "expanded_via_graph": mc,
                                        "related_to": rel_term,
                                    },
                                )
                            )
                            seen.add(entry.chunk_id)

        results.sort(key=lambda r: -r.score)
        return results