from __future__ import annotations

import pytest

from engram.core.types import Chunk, ChunkResult
from engram.core.config import EngramConfig
from engram.retrieve.hash_retriever import HashRetriever
from engram.retrieve.vector_retriever import VectorRetriever
from engram.retrieve.dual import DualRetriever
from engram.retrieve.reranker import ReRanker
from engram.retrieve.context_gate import ContextGate
from engram.retrieve.prefetch import Prefetcher, LRUCache
from engram.retrieve.expansion import QueryExpander
from engram.retrieve.pipeline import RetrievalPipeline
from engram.index.hash_table import MultiHashTable, tokenize_for_concepts
from engram.index.builder import IndexBuilder


def _make_chunks(text: str, doc_id: str = "test") -> list:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        words = tokenize_for_concepts(para)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{i}",
                doc_id=doc_id,
                text=para,
                word_count=len(para.split()),
                page=0,
                section="",
                index=i,
                words=words,
                original_text=para,
            )
        )
    return chunks


def _build_index(text: str):
    chunks = _make_chunks(text)
    config = EngramConfig(max_concepts=20)
    builder = IndexBuilder(config)
    state = builder.build(chunks, text)
    return state


def _make_chunk_results(count: int = 5) -> list:
    results = []
    for i in range(count):
        chunk = Chunk(
            chunk_id=f"c{i}",
            doc_id="d",
            text=f"This is chunk number {i} about topic {i}.",
            word_count=8,
            page=0,
            section="",
            index=i,
            words=["chunk", str(i), "topic"],
        )
        results.append(
            ChunkResult(
                chunk=chunk,
                score=0.9 - i * 0.1,
                source="hash",
                hash_hits=3 - i,
            )
        )
    return results


class TestLRUCache:

    def test_put_and_get(self):
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_eviction(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("c") == 3

    def test_hit_rate(self):
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.get("a")
        cache.get("a")
        cache.get("missing")
        assert cache.hit_rate() > 0

    def test_has(self):
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        assert cache.has("a")
        assert not cache.has("b")

    def test_size(self):
        cache = LRUCache(max_size=5)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 2

    def test_clear(self):
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.clear()
        assert cache.size() == 0
        assert cache.hits == 0

    def test_stats(self):
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.get("a")
        stats = cache.get_stats()
        assert "capacity" in stats
        assert "hits" in stats
        assert "hit_rate" in stats


class TestHashRetriever:

    def test_retrieve(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()

        results = retriever.retrieve(
            "machine learning",
            state.hash_tables,
            state.chunks,
            top_k=5,
        )
        assert len(results) > 0
        for r in results:
            assert r.source == "hash"
            assert r.score > 0

    def test_retrieve_exact_concept(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()

        results = retriever.retrieve(
            "neural network",
            state.hash_tables,
            state.chunks,
            top_k=5,
        )
        assert len(results) > 0

    def test_retrieve_no_match(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()

        results = retriever.retrieve(
            "quantum chromodynamics xyz",
            state.hash_tables,
            state.chunks,
            top_k=5,
        )
        assert len(results) == 0

    def test_retrieve_empty_tables(self):
        retriever = HashRetriever()
        results = retriever.retrieve("test", None, [], top_k=5)
        assert results == []

    def test_results_sorted_by_score(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()

        results = retriever.retrieve(
            "training",
            state.hash_tables,
            state.chunks,
            top_k=10,
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestDualRetriever:

    def test_merge_both_sources(self):
        retriever = DualRetriever()

        hash_results = _make_chunk_results(3)
        for r in hash_results:
            r.source = "hash"

        vector_results = _make_chunk_results(3)
        for r in vector_results:
            r.source = "vector"
            r.vector_sim = r.score

        all_results = hash_results + vector_results
        merged = retriever.merge(all_results)

        assert len(merged) > 0
        for r in merged:
            assert r.source == "dual"

    def test_merge_boosts_overlap(self):
        retriever = DualRetriever()

        chunk = Chunk(
            chunk_id="c0", doc_id="d", text="test",
            word_count=1, page=0, section="", index=0, words=["test"],
        )

        hash_r = ChunkResult(chunk=chunk, score=0.6, source="hash", hash_hits=3)
        vector_r = ChunkResult(chunk=chunk, score=0.7, source="vector", vector_sim=0.7)

        merged = retriever.merge([hash_r, vector_r])
        assert len(merged) == 1
        assert merged[0].score > 0.6

    def test_merge_hash_only(self):
        retriever = DualRetriever()
        hash_results = _make_chunk_results(3)
        for r in hash_results:
            r.source = "hash"

        merged = retriever.merge(hash_results)
        assert len(merged) == 3

    def test_merge_empty(self):
        retriever = DualRetriever()
        merged = retriever.merge([])
        assert merged == []


class TestReRanker:

    def test_rerank(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()
        raw = retriever.retrieve(
            "neural network",
            state.hash_tables,
            state.chunks,
            top_k=10,
        )

        reranker = ReRanker()
        ranked = reranker.rerank(raw, "neural network", chunks=state.chunks)

        assert len(ranked) > 0
        for r in ranked:
            assert r.rerank_score >= 0

    def test_rerank_sorted(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()
        raw = retriever.retrieve(
            "training",
            state.hash_tables,
            state.chunks,
            top_k=10,
        )

        reranker = ReRanker()
        ranked = reranker.rerank(raw, "training", chunks=state.chunks)

        if len(ranked) > 1:
            for i in range(len(ranked) - 1):
                assert ranked[i].rerank_score >= ranked[i + 1].rerank_score

    def test_rerank_with_concept(self, sample_text):
        state = _build_index(sample_text)
        retriever = HashRetriever()
        raw = retriever.retrieve(
            "machine learning",
            state.hash_tables,
            state.chunks,
            top_k=10,
        )

        concept = {"term": "machine learning", "chunk_ids": ["chunk_0"]}
        reranker = ReRanker()
        ranked = reranker.rerank(
            raw, "machine learning", concept=concept, chunks=state.chunks
        )

        assert len(ranked) > 0

    def test_rerank_empty(self):
        reranker = ReRanker()
        ranked = reranker.rerank([], "test")
        assert ranked == []

    def test_diversity_applied(self):
        reranker = ReRanker()
        results = _make_chunk_results(5)
        ranked = reranker.rerank(results, "test topic", chunks=[r.chunk for r in results])
        assert len(ranked) <= len(results)


class TestContextGate:

    def test_gate_filters(self):
        gate = ContextGate(EngramConfig(gate_min_chunks=2, gate_max_chunks=5))
        results = _make_chunk_results(10)

        gated = gate.apply(results)
        assert len(gated) <= 5
        assert len(gated) >= 2

    def test_gate_respects_threshold(self):
        gate = ContextGate(EngramConfig(gate_relevance_threshold=0.5))
        results = _make_chunk_results(10)
        for r in results:
            r.score = 0.3

        gated = gate.apply(results)
        assert len(gated) >= 2

    def test_gate_empty(self):
        gate = ContextGate()
        gated = gate.apply([])
        assert gated == []

    def test_decide_with_relevance(self):
        gate = ContextGate(EngramConfig(gate_min_chunks=2, gate_max_chunks=7))
        results = _make_chunk_results(10)

        gated = gate.decide(results, relevance_score=0.1)
        assert len(gated) <= 7

        gated = gate.decide(results, relevance_score=0.9)
        assert len(gated) <= 4

    def test_get_max_min(self):
        gate = ContextGate(EngramConfig(gate_min_chunks=3, gate_max_chunks=8))
        assert gate.get_min_chunks() == 3
        assert gate.get_max_chunks() == 8


class TestPrefetcher:

    def test_predict_next(self):
        prefetcher = Prefetcher(cache_size=5, predict_count=3)
        concepts = [
            {"term": "a", "score": 10},
            {"term": "b", "score": 20},
            {"term": "c", "score": 15},
            {"term": "d", "score": 5},
        ]
        predicted = prefetcher.predict_next(concepts, count=2)
        assert len(predicted) == 2
        assert "b" in predicted

    def test_cache_hit(self):
        prefetcher = Prefetcher(cache_size=5)
        prefetcher.put("query", ["result"])
        assert prefetcher.has("query")
        assert prefetcher.get("query") == ["result"]

    def test_cache_miss(self):
        prefetcher = Prefetcher(cache_size=5)
        assert not prefetcher.has("missing")
        assert prefetcher.get("missing") is None

    def test_record_access(self):
        prefetcher = Prefetcher(cache_size=5)
        prefetcher.record_access("q1")
        prefetcher.record_access("q2")
        stats = prefetcher.get_stats()
        assert stats["history_length"] == 2

    def test_stats(self):
        prefetcher = Prefetcher(cache_size=5)
        prefetcher.put("q1", "r1")
        prefetcher.get("q1")
        prefetcher.get("missing")
        stats = prefetcher.get_stats()
        assert stats["total_hits"] >= 1
        assert stats["total_misses"] >= 1


class TestQueryExpander:

    def test_expand_sufficient(self):
        expander = QueryExpander()
        results = _make_chunk_results(5)
        expanded = expander.expand(
            "test query", results, None, [], target=3
        )
        assert len(expanded) == 5

    def test_expand_empty(self, sample_text):
        state = _build_index(sample_text)
        expander = QueryExpander()

        expanded = expander.expand(
            "neural network architecture",
            [],
            state.hash_tables,
            state.chunks,
            target=3,
        )
        assert len(expanded) > 0


class TestRetrievalPipeline:

    def test_retrieve_hash(self, sample_text):
        state = _build_index(sample_text)
        config = EngramConfig(retrieval_method="hash", final_top_k=3)
        pipeline = RetrievalPipeline(config)

        results = pipeline.retrieve("machine learning", state, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_returns_chunks(self, sample_text):
        state = _build_index(sample_text)
        config = EngramConfig(retrieval_method="hash", final_top_k=5)
        pipeline = RetrievalPipeline(config)

        results = pipeline.retrieve("neural network", state, top_k=5)
        for r in results:
            assert isinstance(r, ChunkResult)
            assert r.chunk.text.strip() != ""

    def test_retrieve_empty_state(self):
        from engram.core.types import IndexState
        empty_state = IndexState(
            doc_id="empty", chunks=[], concepts=[],
            hash_tables=None, vector_store=None,
            concept_graph=None, ngram_index={},
        )
        pipeline = RetrievalPipeline()
        results = pipeline.retrieve("test", empty_state)
        assert results == []

    def test_pipeline_stats(self):
        pipeline = RetrievalPipeline()
        stats = pipeline.get_stats()
        assert "prefetch" in stats
        assert "method" in stats

    def test_caching(self, sample_text):
        state = _build_index(sample_text)
        config = EngramConfig(retrieval_method="hash", final_top_k=3)
        pipeline = RetrievalPipeline(config)

        r1 = pipeline.retrieve("machine learning", state, top_k=3)
        r2 = pipeline.retrieve("machine learning", state, top_k=3)
        assert len(r1) == len(r2)