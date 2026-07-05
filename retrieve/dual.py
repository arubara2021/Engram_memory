from __future__ import annotations

from typing import Any, Dict, List

from ..core.types import ChunkResult


class DualRetriever:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._hash_weight = getattr(config, "hash_weight", 0.6) if config else 0.6
        self._vector_weight = getattr(config, "vector_weight", 0.4) if config else 0.4

    def merge(self, results: List[ChunkResult]) -> List[ChunkResult]:
        chunk_scores: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cid = r.chunk.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk_result": r,
                    "hash_score": 0.0,
                    "vector_score": 0.0,
                    "hash_hits": 0,
                    "found_by_hash": False,
                    "found_by_vector": False,
                }

            entry = chunk_scores[cid]
            if r.source == "hash":
                entry["hash_score"] = max(entry["hash_score"], r.score)
                entry["hash_hits"] = max(entry["hash_hits"], r.hash_hits)
                entry["found_by_hash"] = True
            elif r.source == "vector":
                entry["vector_score"] = max(entry["vector_score"], r.score)
                entry["found_by_vector"] = True

            if r.score > entry["chunk_result"].score:
                entry["chunk_result"] = r

        merged: List[ChunkResult] = []

        for cid, data in chunk_scores.items():
            hash_s = data["hash_score"]
            vector_s = data["vector_score"]
            found_both = data["found_by_hash"] and data["found_by_vector"]

            if found_both:
                combined = (
                    self._hash_weight * hash_s
                    + self._vector_weight * vector_s
                ) * 1.2
            elif data["found_by_hash"]:
                combined = hash_s * self._hash_weight
            else:
                combined = vector_s * self._vector_weight

            combined = min(combined, 1.0)

            original = data["chunk_result"]
            merged.append(
                ChunkResult(
                    chunk=original.chunk,
                    score=combined,
                    source="dual",
                    hash_hits=data["hash_hits"],
                    vector_sim=vector_s,
                    rerank_score=0.0,
                    match_details={
                        "hash_score": hash_s,
                        "vector_score": vector_s,
                        "found_by_both": found_both,
                        "hash_weight": self._hash_weight,
                        "vector_weight": self._vector_weight,
                    },
                )
            )

        merged.sort(key=lambda r: -r.score)
        return merged