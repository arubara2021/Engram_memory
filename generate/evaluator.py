from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.types import ChunkResult


class ResponseEvaluator:

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        response: str,
        source_chunks: List[ChunkResult],
    ) -> Dict[str, Any]:
        if not response or not source_chunks:
            return {
                "grounding_score": 0.0,
                "source_coverage": 0.0,
                "hallucination_flags": [],
                "is_grounded": False,
                "details": {},
            }

        response_sentences = self._split_sentences(response)
        source_text = " ".join(c.chunk.text for c in source_chunks)
        source_words = set(self._tokenize(source_text))

        grounded_count = 0
        hallucination_flags: List[str] = []

        for sent in response_sentences:
            sent_words = set(self._tokenize(sent))
            if not sent_words:
                continue

            overlap = len(sent_words & source_words)
            coverage = overlap / max(len(sent_words), 1)

            if coverage >= 0.3:
                grounded_count += 1
            elif len(sent_words) > 5:
                hallucination_flags.append(sent.strip())

        grounding_score = grounded_count / max(len(response_sentences), 1)

        source_coverage = self._compute_source_coverage(response, source_chunks)

        has_uncertain = self._detect_uncertainty(response)

        return {
            "grounding_score": round(grounding_score, 3),
            "source_coverage": round(source_coverage, 3),
            "hallucination_flags": hallucination_flags[:10],
            "is_grounded": grounding_score >= 0.5,
            "has_uncertainty_language": has_uncertain,
            "details": {
                "response_sentence_count": len(response_sentences),
                "grounded_sentences": grounded_count,
                "potential_hallucinations": len(hallucination_flags),
                "source_chunk_count": len(source_chunks),
            },
        }

    def compute_confidence(
        self,
        chunks: List[ChunkResult],
        final_top_k: int = 5,
    ) -> float:
        if not chunks:
            return 0.0

        scores = [c.score for c in chunks]
        avg_score = sum(scores) / len(scores)
        coverage = min(len(chunks) / max(final_top_k, 1), 1.0)

        confidence = avg_score * coverage
        return round(min(confidence, 1.0), 3)

    def check_source_grounding(
        self,
        claim: str,
        source_chunks: List[ChunkResult],
    ) -> Dict[str, Any]:
        claim_words = set(self._tokenize(claim))
        if not claim_words:
            return {"grounded": False, "supporting_chunks": [], "best_overlap": 0.0}

        supporting_chunks: List[Dict[str, Any]] = []
        best_overlap = 0.0

        for cr in source_chunks:
            chunk_words = set(self._tokenize(cr.chunk.text))
            if not chunk_words:
                continue

            overlap = len(claim_words & chunk_words)
            overlap_ratio = overlap / max(len(claim_words), 1)

            if overlap_ratio > 0.2:
                supporting_chunks.append({
                    "chunk_id": cr.chunk.chunk_id,
                    "overlap_ratio": round(overlap_ratio, 3),
                    "overlap_words": overlap,
                    "text_preview": cr.chunk.text[:200],
                })

            best_overlap = max(best_overlap, overlap_ratio)

        return {
            "grounded": best_overlap >= 0.3,
            "supporting_chunks": supporting_chunks,
            "best_overlap": round(best_overlap, 3),
        }

    def _compute_source_coverage(
        self, response: str, source_chunks: List[ChunkResult]
    ) -> float:
        if not source_chunks:
            return 0.0

        source_text = " ".join(c.chunk.text for c in source_chunks)
        source_sentences = self._split_sentences(source_text)
        if not source_sentences:
            return 0.0

        response_words = set(self._tokenize(response))
        if not response_words:
            return 0.0

        covered = 0
        for sent in source_sentences:
            sent_words = set(self._tokenize(sent))
            if not sent_words:
                continue
            overlap = len(sent_words & response_words)
            if overlap / max(len(sent_words), 1) > 0.2:
                covered += 1

        return covered / max(len(source_sentences), 1)

    def _detect_uncertainty(self, text: str) -> bool:
        uncertainty_phrases = [
            "not in the provided",
            "not covered in",
            "not mentioned in",
            "i don't have",
            "cannot be determined",
            "the source does not",
            "the documents do not",
            "this information is not",
            "based on the provided",
            "according to the source",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in uncertainty_phrases)

    def _split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return [w for w in text.split() if len(w) > 1]