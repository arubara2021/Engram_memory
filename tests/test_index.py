from __future__ import annotations

import pytest

from engram.core.types import Chunk
from engram.index.hash_table import (
    MultiHashTable,
    hash_fnv1a,
    hash_murmur,
    hash_djb2,
    hash_all,
    calculate_table_size,
    extract_ngrams_from_tokens,
    tokenize_for_concepts,
    normalize_text,
    is_url_noise,
    COMMON_WORDS,
    NOISE_TERMS,
)
from engram.index.concept import ConceptExtractor, is_technical_term, filter_concept_noise
from engram.index.ngram import NgramIndexer
from engram.index.graph import ConceptGraph
from engram.index.builder import IndexBuilder
from engram.index.cache import ApproximateCache
from engram.core.config import EngramConfig


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


class TestHashFunctions:

    def test_fnv1a_deterministic(self):
        assert hash_fnv1a("test", 4096) == hash_fnv1a("test", 4096)

    def test_murmur_deterministic(self):
        assert hash_murmur("test", 4096) == hash_murmur("test", 4096)

    def test_djb2_deterministic(self):
        assert hash_djb2("test", 4096) == hash_djb2("test", 4096)

    def test_hash_in_range(self):
        for table_size in [100, 1000, 4096, 65536]:
            h = hash_fnv1a("hello world", table_size)
            assert 0 <= h < table_size

    def test_hash_all_returns_three(self):
        result = hash_all("test", 4096)
        assert len(result) == 3
        for h in result:
            assert 0 <= h < 4096

    def test_different_strings_different_hashes(self):
        h1 = hash_fnv1a("alpha", 65536)
        h2 = hash_fnv1a("beta", 65536)
        assert h1 != h2

    def test_calculate_table_size(self):
        size = calculate_table_size(1000, 0.35)
        assert size >= 1000 / 0.35
        assert size >= 4096
        assert (size & (size - 1)) == 0

    def test_empty_string(self):
        h = hash_fnv1a("", 4096)
        assert isinstance(h, int)
        assert 0 <= h < 4096


class TestMultiHashTable:

    def test_build_and_lookup(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        build_time = table.build_from_chunks(chunks)

        assert build_time > 0
        assert table.total_insertions > 0
        assert table.table_size >= 4096

    def test_lookup_by_ngram(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        results = table.lookup_by_ngram("machine learning")
        assert len(results) > 0

    def test_lookup_concept(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        concepts = [{"term": "neural network", "chunk_ids": ["chunk_1"], "score": 50, "definition": "", "primary_chunk_id": "chunk_1"}]
        table.build_from_chunks(chunks, concepts)

        results = table.lookup("neural network")
        assert len(results) > 0

    def test_lookup_nonexistent(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        results = table.lookup("xyznonexistent123")
        assert len(results) == 0

    def test_serialize_deserialize(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        serialized = table.serialize()
        assert len(serialized) > 0

        restored = MultiHashTable.deserialize(serialized)
        assert restored.table_size == table.table_size
        assert restored.total_insertions == table.total_insertions

    def test_stats(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        stats = table.get_stats()
        assert stats.total_entries > 0
        assert stats.total_unique_ngrams > 0
        assert stats.table_size >= 4096
        assert 0 < stats.load_factor < 1

    def test_concept_registration(self):
        table = MultiHashTable()
        table.register_concept("test_concept", ["test", "concept"], ["c0", "c1"])
        assert "test_concept" in table.concept_table
        assert table.concept_table["test_concept"].primary_chunk_id == "c0"


class TestConceptExtractor:

    def test_extract_concepts(self, sample_text):
        chunks = _make_chunks(sample_text)
        config = EngramConfig(max_concepts=20)
        extractor = ConceptExtractor(config)
        concepts = extractor.extract(chunks, sample_text)

        assert len(concepts) > 0
        assert len(concepts) <= 20

        for c in concepts:
            assert c.score > 0
            assert c.frequency >= 1
            assert len(c.chunk_ids) > 0

    def test_concepts_have_labels(self, sample_text):
        chunks = _make_chunks(sample_text)
        extractor = ConceptExtractor()
        concepts = extractor.extract(chunks, sample_text)

        for c in concepts:
            assert c.label.strip() != ""

    def test_no_noise_in_concepts(self, sample_text):
        chunks = _make_chunks(sample_text)
        extractor = ConceptExtractor()
        concepts = extractor.extract(chunks, sample_text)

        for c in concepts[:10]:
            assert not is_url_noise(c.label)

    def test_is_technical_term(self):
        assert is_technical_term("algorithm")
        assert is_technical_term("algorithm")
        assert not is_technical_term("the")
        assert not is_technical_term("a")

    def test_filter_concept_noise(self):
        assert filter_concept_noise({"term": "attention mechanism"})
        assert not filter_concept_noise({"term": "http"})
        assert not filter_concept_noise({"term": ""})


class TestNgramIndexer:

    def test_build_and_lookup(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        config = EngramConfig(max_concepts=10)
        extractor = ConceptExtractor(config)
        concepts = extractor.extract(chunks, sample_text)

        indexer = NgramIndexer()
        ngram_index = indexer.build_index(chunks, concepts, table)

        assert len(ngram_index) > 0

    def test_lookup(self, sample_text):
        chunks = _make_chunks(sample_text)
        table = MultiHashTable()
        table.build_from_chunks(chunks)

        indexer = NgramIndexer()
        results = indexer.lookup("neural network", table)

        assert len(results) > 0


class TestConceptGraph:

    def test_build_graph(self, sample_text):
        chunks = _make_chunks(sample_text)
        concepts = [
            {"term": "machine learning", "chunk_ids": ["chunk_0"], "score": 50},
            {"term": "neural network", "chunk_ids": ["chunk_1"], "score": 45},
            {"term": "deep learning", "chunk_ids": ["chunk_1", "chunk_2"], "score": 40},
        ]

        graph = ConceptGraph()
        graph.build(concepts, chunks)

        assert len(graph.edges) >= 0

    def test_get_related(self):
        graph = ConceptGraph()
        graph.adjacency["a"] = {"b", "c"}
        graph.adjacency["b"] = {"a"}
        graph.adjacency["c"] = {"a"}

        related = graph.get_related("a", max_depth=1)
        assert "b" in related
        assert "c" in related

    def test_serialize_deserialize(self, sample_text):
        chunks = _make_chunks(sample_text)
        concepts = [
            {"term": "machine learning", "chunk_ids": ["chunk_0"], "score": 50},
            {"term": "neural network", "chunk_ids": ["chunk_1"], "score": 45},
        ]

        graph = ConceptGraph()
        graph.build(concepts, chunks)

        serialized = graph.serialize()
        restored = ConceptGraph.deserialize(serialized)

        assert len(restored.edges) == len(graph.edges)


class TestIndexBuilder:

    def test_build_full_index(self, sample_text):
        chunks = _make_chunks(sample_text)
        config = EngramConfig(max_concepts=20)
        builder = IndexBuilder(config)

        state = builder.build(chunks, sample_text)

        assert len(state.chunks) == len(chunks)
        assert len(state.concepts) > 0
        assert state.hash_tables is not None
        assert len(state.ngram_index) > 0

    def test_concepts_have_primary_chunks(self, sample_text):
        chunks = _make_chunks(sample_text)
        builder = IndexBuilder()
        state = builder.build(chunks, sample_text)

        for c in state.concepts:
            if c.chunk_ids:
                assert c.primary_chunk_id is not None


class TestApproximateCache:

    def test_put_and_get(self):
        cache = ApproximateCache(max_size=5)
        cache.put("query one", ["result_a", "result_b"])

        result = cache.get("query one")
        assert result == ["result_a", "result_b"]

    def test_exact_match(self):
        cache = ApproximateCache(max_size=5)
        cache.put("what is machine learning", ["c1", "c2"])

        assert cache.get("what is machine learning") == ["c1", "c2"]

    def test_similar_query(self):
        cache = ApproximateCache(max_size=5, similarity_threshold=0.5)
        cache.put("what is machine learning", ["c1", "c2"])

        result = cache.get("what is the machine learning")
        assert result is not None

    def test_miss(self):
        cache = ApproximateCache(max_size=5, similarity_threshold=0.99)
        cache.put("completely different topic", ["c1"])

        result = cache.get("another unrelated query about something else")
        assert result is None

    def test_eviction(self):
        cache = ApproximateCache(max_size=2)
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.put("q3", "r3")

        assert cache.size() <= 2
        assert cache.has("q3")

    def test_stats(self):
        cache = ApproximateCache(max_size=5)
        cache.put("q1", "r1")
        cache.get("q1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["size"] >= 1

    def test_clear(self):
        cache = ApproximateCache(max_size=5)
        cache.put("q1", "r1")
        cache.clear()

        assert cache.size() == 0
        assert cache.get_stats()["hits"] == 0
