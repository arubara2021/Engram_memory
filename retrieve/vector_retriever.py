from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.types import Chunk, ChunkResult


class VectorRetriever:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._threshold = getattr(config, "similarity_threshold", 0.3) if config else 0.3
        self._top_k = getattr(config, "rerank_top_k", 20) if config else 20

    def retrieve(
        self,
        query: str,
        vector_store: Any,
        chunks: List[Chunk],
        embedding_pipeline: Any = None,
        top_k: int = 20,
    ) -> List[ChunkResult]:
        if not vector_store or not chunks:
            return []

        query_vector = None
        if embedding_pipeline and embedding_pipeline.is_available():
            query_vector = embedding_pipeline.embed_query(query)

        if not query_vector:
            return []

        raw_results = vector_store.search(
            query_vector, top_k=top_k * 2, threshold=self._threshold
        )

        if not raw_results:
            return []

        chunk_map = {c.chunk_id: c for c in chunks}
        results: List[ChunkResult] = []

        for chunk_id, similarity in raw_results:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue

            results.append(
                ChunkResult(
                    chunk=chunk,
                    score=similarity,
                    source="vector",
                    hash_hits=0,
                    vector_sim=similarity,
                    rerank_score=0.0,
                    match_details={
                        "cosine_similarity": similarity,
                        "query_vector_dim": len(query_vector),
                    },
                )
            )

        results.sort(key=lambda r: -r.score)
        return results[:top_k]