from __future__ import annotations

import math

import pytest

from engram.embed.store import VectorStore
from engram.embed.local import LocalEmbedder
from engram.embed.remote import RemoteEmbedder
from engram.embed.pipeline import EmbeddingPipeline


class TestVectorStore:

    def test_add_and_get(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        assert store.has("c1")
        assert store.get("c1") == [1.0, 0.0, 0.0]

    def test_add_batch(self):
        store = VectorStore(dimension=3)
        store.add_batch(
            ["c1", "c2", "c3"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        assert store.size() == 3

    def test_search_cosine_similarity(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.add("c2", [0.0, 1.0, 0.0])
        store.add("c3", [0.7, 0.7, 0.0])

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0][0] == "c1"
        assert results[0][1] > results[1][1]

    def test_search_with_threshold(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.add("c2", [0.0, 1.0, 0.0])

        results = store.search([1.0, 0.0, 0.0], top_k=10, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == "c1"

    def test_remove(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.remove("c1")
        assert not store.has("c1")
        assert store.size() == 0

    def test_search_empty(self):
        store = VectorStore(dimension=3)
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    def test_metadata(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0], {"doc_id": "d1"})
        meta = store.get_metadata("c1")
        assert meta is not None
        assert meta["doc_id"] == "d1"

    def test_clear(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.add("c2", [0.0, 1.0, 0.0])
        store.clear()
        assert store.size() == 0

    def test_serialize_json(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0], {"doc_id": "d1"})
        store.add("c2", [0.0, 1.0, 0.0])

        data = store.serialize()
        restored = VectorStore.deserialize(data)

        assert restored.size() == 2
        assert restored.get("c1") == [1.0, 0.0, 0.0]
        assert restored.get_metadata("c1") == {"doc_id": "d1"}

    def test_serialize_binary(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.add("c2", [0.0, 1.0, 0.0])

        data = store.serialize_binary()
        restored = VectorStore.deserialize_binary(data)

        assert restored.size() == 2
        assert restored.dimension == 3

    def test_identical_vectors(self):
        store = VectorStore(dimension=3)
        store.add("c1", [1.0, 0.0, 0.0])
        store.add("c2", [1.0, 0.0, 0.0])

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert abs(results[0][1] - results[1][1]) < 0.0001

    def test_orthogonal_vectors(self):
        store = VectorStore(dimension=2)
        store.add("c1", [1.0, 0.0])
        store.add("c2", [0.0, 1.0])

        results = store.search([1.0, 0.0], top_k=2)
        assert results[0][0] == "c1"
        assert abs(results[0][1] - 1.0) < 0.0001
        assert abs(results[1][1]) < 0.0001

    def test_get_all_ids(self):
        store = VectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("b", [0.0, 1.0])
        ids = store.get_all_ids()
        assert set(ids) == {"a", "b"}


class TestLocalEmbedder:

    def test_fallback_tfidf(self):
        embedder = LocalEmbedder()
        if embedder.load():
            vectors = embedder.embed(["hello world", "test document"])
            assert len(vectors) == 2
            assert len(vectors[0]) > 0
            assert len(vectors[0]) == len(vectors[1])

    def test_single_embed(self):
        embedder = LocalEmbedder()
        if embedder.load():
            vector = embedder.embed_single("test text")
            assert len(vector) > 0

    def test_dimension(self):
        embedder = LocalEmbedder(dimension=128)
        assert embedder.dimension == 128

    def test_is_available(self):
        embedder = LocalEmbedder()
        assert isinstance(embedder.is_available(), bool)


class TestRemoteEmbedder:

    def test_not_available_without_config(self):
        embedder = RemoteEmbedder()
        assert embedder.is_available() is False

    def test_available_with_config(self):
        embedder = RemoteEmbedder(api_url="https://api.example.com", api_key="sk-test")
        assert embedder.is_available() is True


class TestEmbeddingPipeline:

    def test_default_not_available(self):
        pipeline = EmbeddingPipeline()
        assert not pipeline.is_available()

    def test_with_config_disabled(self):
        from engram.core.config import EngramConfig
        config = EngramConfig(embedding_enabled=False)
        pipeline = EmbeddingPipeline(config)
        assert not pipeline.is_available()

    def test_set_backend(self):
        embedder = LocalEmbedder()
        if embedder.load():
            pipeline = EmbeddingPipeline()
            pipeline.set_backend(embedder)
            assert pipeline.is_available()

    def test_get_stats(self):
        pipeline = EmbeddingPipeline()
        stats = pipeline.get_stats()
        assert "available" in stats
        assert "dimension" in stats
        assert "stored_vectors" in stats

    def test_embed_and_search(self):
        embedder = LocalEmbedder()
        if embedder.load():
            pipeline = EmbeddingPipeline()
            pipeline.set_backend(embedder)

            from engram.core.types import Chunk
            chunks = [
                Chunk(chunk_id="c0", doc_id="d", text="Machine learning algorithms",
                      word_count=3, page=0, section="", index=0, words=["machine", "learning", "algorithms"]),
                Chunk(chunk_id="c1", doc_id="d", text="Neural network architecture",
                      word_count=3, page=0, section="", index=1, words=["neural", "network", "architecture"]),
            ]

            updated = pipeline.embed_chunks(chunks)
            assert updated[0].embedding is not None
            assert updated[1].embedding is not None
            assert pipeline.store.size() == 2

            results = pipeline.search("machine learning", top_k=2)
            assert len(results) > 0