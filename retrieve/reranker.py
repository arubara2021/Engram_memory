from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from ..core.types import Chunk, ChunkResult

from ..index.hash_table import (
    COMMON_WORDS,
    NOISE_TERMS,
    extract_ngrams_from_tokens,
    normalize_text,
    tokenize_for_concepts,
)

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "with", "and", "or", "but", "how", "what",
    "why", "when", "where", "do", "does", "did", "can", "could",
    "would", "should", "this", "that", "it", "its", "be", "been",
    "being", "have", "has", "had", "about", "up", "down", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "from", "by", "as", "into", "through", "during", "before", "after",
}


class ReRanker:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._coverage_weight = 0.30
        self._density_weight = 0.25
        self._alignment_weight = 0.20
        self._position_weight = 0.08
        self._specificity_weight = 0.12
        self._proximity_weight = 0.05

    def rerank(
        self,
        results: List[ChunkResult],
        query: str,
        concept: Optional[Dict] = None,
        chunks: Optional[List[Chunk]] = None,
    ) -> List[ChunkResult]:
        if not results:
            return []

        query_tokens = tokenize_for_concepts(query)
        query_ngrams: Set[str] = set()
        for ng, _ in extract_ngrams_from_tokens(query_tokens, 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS:
                query_ngrams.add(t)

        if not query_ngrams:
            normalized = normalize_text(query)
            if normalized:
                query_ngrams = {normalized}

        if not query_ngrams:
            for r in results:
                r.rerank_score = r.score
            results.sort(key=lambda r: -r.rerank_score)
            return results

        concept_chunk_ids: Set[str] = set()
        if concept:
            concept_chunk_ids = set(concept.get("chunk_ids", []))

        chunk_map = {c.chunk_id: c for c in chunks} if chunks else {}
        total_chunks = len(chunks) if chunks else 1

        query_content = {
            w for w in query_tokens
            if w not in STOP_WORDS and w not in NOISE_TERMS and len(w) > 2
        }
        query_stems = self._stem_set(query_content)

        for r in results:
            chunk = r.chunk
            if chunk.chunk_id in chunk_map:
                chunk = chunk_map[chunk.chunk_id]

            chunk_ngrams: Set[str] = set()
            for ng, _ in extract_ngrams_from_tokens(chunk.words, 4):
                chunk_ngrams.add(ng)
            for t in chunk.words:
                if t not in NOISE_TERMS:
                    chunk_ngrams.add(t)

            matching = query_ngrams & chunk_ngrams

            chunk_content = {
                w for w in chunk.words
                if w not in STOP_WORDS and w not in NOISE_TERMS and len(w) > 2
            }
            chunk_stems = self._stem_set(chunk_content)

            exact_overlap = (
                len(query_content & chunk_content) / len(query_content)
                if query_content
                else 0.0
            )
            stem_overlap = (
                len(query_stems & chunk_stems) / len(query_stems)
                if query_stems
                else 0.0
            )
            content_overlap = max(exact_overlap, stem_overlap)

            coverage_raw = len(matching) / max(len(query_ngrams), 1)
            coverage_score = coverage_raw * 0.5 + content_overlap * 0.5

            density_score = (
                len(matching) / max(len(chunk_ngrams), 1)
                if chunk_ngrams
                else 0.0
            )

            alignment_score = (
                1.0 if chunk.chunk_id in concept_chunk_ids else 0.0
            )

            position_score = 1.0 / (
                1.0 + chunk.index / max(total_chunks, 1)
            )

            specificity = self._compute_specificity(chunk, query_content)

            proximity = self._compute_proximity(chunk.text, query_content)

            diversity_penalty = self._compute_diversity_penalty(r, results)

            initial = (
                self._coverage_weight * coverage_score
                + self._density_weight * density_score
                + self._alignment_weight * alignment_score
                + self._position_weight * position_score
                + self._specificity_weight * specificity
                + self._proximity_weight * proximity
                - diversity_penalty
            )

            initial = max(initial, 0.0)

            micro = self._micro_differentiator(
                initial, coverage_score, density_score, specificity,
                proximity, position_score, content_overlap,
            )

            r.rerank_score = min(initial + micro, 1.0)
            r.match_details = {
                "coverage": round(coverage_score, 4),
                "density": round(density_score, 4),
                "alignment": round(alignment_score, 4),
                "position": round(position_score, 4),
                "specificity": round(specificity, 4),
                "proximity": round(proximity, 4),
                "content_overlap": round(content_overlap, 4),
                "diversity_penalty": round(diversity_penalty, 4),
                "micro_tiebreaker": round(micro, 6),
                "matching_ngrams": len(matching),
                "total_query_ngrams": len(query_ngrams),
            }

        results.sort(key=lambda r: (-r.rerank_score, r.chunk.index))
        results = self._apply_diversity(results)
        return results

    def _micro_differentiator(
        self,
        score: float,
        coverage: float,
        density: float,
        specificity: float,
        proximity: float,
        position: float,
        content_overlap: float,
    ) -> float:
        micro = 0.0
        micro += (density - int(density * 1000) / 1000) * 0.001
        micro += (proximity - int(proximity * 1000) / 1000) * 0.001
        micro += (specificity - int(specificity * 100) / 100) * 0.001
        micro += (position - int(position * 100) / 100) * 0.0005
        micro += (content_overlap - int(content_overlap * 100) / 100) * 0.0005
        return micro

    def _compute_specificity(
        self, chunk: Chunk, query_content: Set[str]
    ) -> float:
        if not query_content or not chunk.words:
            return 0.0

        text = " ".join(chunk.words).lower()
        total_words = len(chunk.words)
        if total_words == 0:
            return 0.0

        match_count = sum(1 for w in query_content if w in text)

        density = match_count / total_words
        coverage = match_count / len(query_content)

        return min(coverage * 0.6 + density * 40, 1.0)

    def _compute_proximity(
        self, text: str, query_content: Set[str]
    ) -> float:
        if len(query_content) < 2:
            return 0.5

        words = text.lower().split()
        if not words:
            return 0.0

        positions = {}
        for i, w in enumerate(words):
            for qw in query_content:
                if w == qw and qw not in positions:
                    positions[qw] = i

        if len(positions) < 2:
            return 0.0

        pos_list = sorted(positions.values())
        gaps = []
        for i in range(1, len(pos_list)):
            gaps.append(pos_list[i] - pos_list[i - 1])

        avg_gap = sum(gaps) / len(gaps)
        proximity = 1.0 / (1.0 + avg_gap / 10.0)

        found_ratio = len(positions) / len(query_content)
        return proximity * found_ratio

    def _stem_set(self, words: Set[str]) -> Set[str]:
        stems = set()
        suffixes = [
            "ting", "ing", "tion", "sion", "ment", "ness", "able",
            "ible", "ful", "ous", "ive", "ory", "ize", "ise", "ify",
            "ate", "ent", "ant", "est", "ers", "ies", "ed", "er", "es", "ly",
        ]
        for w in words:
            stems.add(w)
            if len(w) > 3:
                for s in suffixes:
                    if w.endswith(s) and len(w) - len(s) >= 3:
                        stems.add(w[:-len(s)])
                        break
        return stems

    def _compute_diversity_penalty(
        self, current: ChunkResult, all_results: List[ChunkResult]
    ) -> float:
        current_words = set(current.chunk.words[:20])
        if not current_words:
            return 0.0

        max_overlap = 0.0
        for other in all_results:
            if other.chunk.chunk_id == current.chunk.chunk_id:
                continue
            other_words = set(other.chunk.words[:20])
            if not other_words:
                continue
            overlap = len(current_words & other_words) / max(
                len(current_words | other_words), 1
            )
            max_overlap = max(max_overlap, overlap)

        if max_overlap > 0.8:
            return 0.2
        if max_overlap > 0.6:
            return 0.1
        return 0.0

    def _apply_diversity(
        self, results: List[ChunkResult]
    ) -> List[ChunkResult]:
        if len(results) <= 1:
            return results

        selected: List[ChunkResult] = []
        remaining = list(results)

        while remaining:
            best = remaining.pop(0)
            selected.append(best)

            best_words = set(best.chunk.words[:30])
            new_remaining: List[ChunkResult] = []
            for r in remaining:
                r_words = set(r.chunk.words[:30])
                if best_words and r_words:
                    overlap = len(best_words & r_words) / max(
                        len(best_words | r_words), 1
                    )
                    if overlap < 0.85:
                        new_remaining.append(r)
                else:
                    new_remaining.append(r)
            remaining = new_remaining

        return selected
