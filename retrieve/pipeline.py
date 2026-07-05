from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..core.types import Chunk, ChunkResult, IndexState

from .hash_retriever import HashRetriever
from .vector_retriever import VectorRetriever
from .dual import DualRetriever
from .reranker import ReRanker
from .context_gate import ContextGate
from .prefetch import Prefetcher
from .expansion import QueryExpander


class RetrievalPipeline:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.hash_retriever = HashRetriever(config)
        self.vector_retriever = VectorRetriever(config)
        self.dual_retriever = DualRetriever(config)
        self.reranker = ReRanker(config)
        self.context_gate = ContextGate(config)
        self.prefetcher = Prefetcher(
            cache_size=getattr(config, "prefetch_cache_size", 10) if config else 10,
            predict_count=getattr(config, "prefetch_predict_count", 3) if config else 3,
        )
        self.expander = QueryExpander(config)
        self._embedding_pipeline: Any = None

    def set_embedding_pipeline(self, pipeline: Any) -> None:
        self._embedding_pipeline = pipeline

    def retrieve(
        self,
        query: str,
        state: IndexState,
        doc_id: str = None,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        concepts: Optional[List[Dict]] = None,
        concept_graph: Any = None,
    ) -> List[ChunkResult]:
        if not state.chunks:
            return []

        effective_top_k = top_k or (
            getattr(self.config, "final_top_k", 5) if self.config else 5
        )
        effective_method = method or (
            getattr(self.config, "retrieval_method", "hash") if self.config else "hash"
        )
        rerank_top_k = (
            getattr(self.config, "rerank_top_k", 20) if self.config else 20
        )

        cached = self.prefetcher.get(query)
        if cached is not None:
            self.prefetcher.record_access(query)
            return cached[:effective_top_k]

        results: List[ChunkResult] = []

        if effective_method in ("hash", "dual"):
            hash_results = self._hash_retrieve(query, state, rerank_top_k)
            results.extend(hash_results)

        if effective_method in ("vector", "dual"):
            vector_results = self._vector_retrieve(query, state, rerank_top_k)
            results.extend(vector_results)

        if effective_method == "dual" and results:
            has_hash = any(r.source == "hash" for r in results)
            has_vector = any(r.source == "vector" for r in results)
            if has_hash and has_vector:
                results = self.dual_retriever.merge(results)

        concepts_for_expansion = concepts or [
            {"term": c.label, "chunk_ids": c.chunk_ids, "score": c.score}
            for c in state.concepts
        ]

        if len(results) < 1:
            results = self.expander.expand(
                query,
                results,
                state.hash_tables,
                state.chunks,
                concepts_for_expansion,
                concept_graph or state.concept_graph,
                target=rerank_top_k,
            )

        if effective_method == "hash":
            for r in results:
                if r.source == "hash":
                    r.source = "hash"
                elif r.source == "expansion":
                    r.source = "hash"

        results = self.reranker.rerank(
            results,
            query,
            concept=None,
            chunks=state.chunks,
        )

        results = self.context_gate.apply(results)

        seen: set = set()
        deduped: List[ChunkResult] = []
        for r in results:
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                deduped.append(r)

        final = deduped[:effective_top_k]

        self.prefetcher.put(query, final)
        self.prefetcher.record_access(query)

        self._background_prefetch(state, concepts_for_expansion, concept_graph)

        return final

    def _hash_retrieve(
        self, query: str, state: IndexState, top_k: int
    ) -> List[ChunkResult]:
        if state.hash_tables is None:
            return []
        return self.hash_retriever.retrieve(
            query, state.hash_tables, state.chunks, top_k, doc_id=doc_id
        )

    def _vector_retrieve(
        self, query: str, state: IndexState, top_k: int
    ) -> List[ChunkResult]:
        if state.vector_store is None:
            return []
        if self._embedding_pipeline is None or not self._embedding_pipeline.is_available():
            return []
        return self.vector_retriever.retrieve(
            query,
            state.vector_store,
            state.chunks,
            self._embedding_pipeline,
            top_k,
        )

    def _background_prefetch(
        self,
        state: IndexState,
        concepts: List[Dict],
        concept_graph: Any,
    ) -> None:
        prefetch_enabled = (
            getattr(self.config, "prefetch_enabled", True) if self.config else True
        )
        if not prefetch_enabled:
            return

        try:
            predicted = self.prefetcher.predict_from_history(
                concepts, concept_graph
            )
            for term in predicted:
                if not self.prefetcher.has(term):
                    hash_results = self._hash_retrieve(term, state, 10)
                    if hash_results:
                        self.prefetcher.put(term, hash_results)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "prefetch": self.prefetcher.get_stats(),
            "method": getattr(self.config, "retrieval_method", "hash") if self.config else "hash",
            "embedding_available": (
                self._embedding_pipeline.is_available()
                if self._embedding_pipeline
                else False
            ),
        }