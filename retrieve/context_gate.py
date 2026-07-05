from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.types import ChunkResult

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "with", "and", "or", "but", "how", "what",
    "why", "when", "where", "do", "does", "did", "can", "could",
    "would", "should", "this", "that", "it", "its", "be", "been",
    "being", "have", "has", "had", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "the", "about",
}


def query_specificity(query: str) -> float:
    words = re.sub(r"[^\w\s]", " ", query.lower()).split()
    content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    total = len(words)
    if total == 0:
        return 0.5
    content_ratio = len(content_words) / total
    has_quotes = '"' in query or "'" in query
    has_specific = any(
        phrase in query.lower()
        for phrase in [
            "what is", "define", "explain", "how does",
            "what are", "describe", "difference between",
        ]
    )
    specificity = content_ratio
    if has_quotes:
        specificity += 0.2
    if has_specific:
        specificity += 0.1
    if len(content_words) >= 4:
        specificity += 0.1
    return min(specificity, 1.0)


class ContextGate:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._min_chunks = (
            getattr(config, "gate_min_chunks", 2) if config else 2
        )
        self._max_chunks = (
            getattr(config, "gate_max_chunks", 7) if config else 7
        )
        self._threshold = (
            getattr(config, "gate_relevance_threshold", 0.3)
            if config
            else 0.3
        )

    def apply(self, results: List[ChunkResult]) -> List[ChunkResult]:
        if not results:
            return []

        filtered = [r for r in results if r.score >= self._threshold]

        if len(filtered) < self._min_chunks:
            remaining = [
                r for r in results if r.score < self._threshold
            ]
            remaining.sort(key=lambda r: -r.score)
            filtered.extend(
                remaining[: self._min_chunks - len(filtered)]
            )

        return filtered[: self._max_chunks]

    def decide(
        self,
        results: List[ChunkResult],
        relevance_score: Optional[float] = None,
        query: Optional[str] = None,
    ) -> List[ChunkResult]:
        if not results:
            return []

        specificity = query_specificity(query) if query else 0.5

        score_spread = self._score_spread(results)
        avg_score = self._avg_score(results)

        if specificity > 0.7:
            max_c = max(self._min_chunks, 3)
            threshold = max(self._threshold, avg_score * 0.7)
        elif specificity > 0.4:
            max_c = self._max_chunks
            threshold = self._threshold
        else:
            max_c = min(self._max_chunks + 3, 10)
            threshold = min(self._threshold, 0.15)

        if relevance_score is not None:
            if relevance_score < 0.2:
                max_c = min(max_c + 2, 10)
            elif relevance_score > 0.7:
                max_c = max(self._min_chunks, max_c - 2)

        filtered = [r for r in results if r.score >= threshold]
        if len(filtered) < self._min_chunks:
            remaining = [r for r in results if r.score < threshold]
            remaining.sort(key=lambda r: -r.score)
            filtered.extend(
                remaining[: self._min_chunks - len(filtered)]
            )

        if score_spread < 0.05 and len(filtered) > self._min_chunks + 1:
            top_score = filtered[0].score
            filtered = [
                r for r in filtered if r.score >= top_score - 0.15
            ]
            if len(filtered) < self._min_chunks:
                filtered = results[: self._min_chunks]

        return filtered[:max_c]

    def _score_spread(self, results: List[ChunkResult]) -> float:
        if len(results) < 2:
            return 0.0
        scores = [r.score for r in results]
        return max(scores) - min(scores)

    def _avg_score(self, results: List[ChunkResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def get_max_chunks(self) -> int:
        return self._max_chunks

    def get_min_chunks(self) -> int:
        return self._min_chunks
