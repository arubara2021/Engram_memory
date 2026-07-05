from __future__ import annotations

import os

import pytest

from engram.store.lru import LRUCache
from engram.store.serializer import Serializer
from engram.store.filesystem import FileStore
from engram.store.project import ProjectStore
from engram.core.types import Chunk, Concept, IndexState


def _make_test_state() -> IndexState:
    chunks = [
        Chunk(
            chunk_id=f"chunk_{i}",
            doc_id="test_doc",
            text=f"This is chunk number {i} with some content.",
            word_count=8,
            page=i,
            section=f"Section {i}",
            index=i,
            words=["chunk", "number", str(i), "content"],
            original_text=f"This is chunk number {i} with some content.",
        )
        for i in range(3)
    ]

    concepts = [
        Concept(
            concept_id="test_concept",
            label="test concept",
            ngrams=["test", "concept", "test concept"],
            frequency=5,
            chunk_ids=["chunk_0", "chunk_1"],
            score=45.0,
            tags=["D"],
            primary_chunk_id="chunk_0",
        ),
    ]

    return IndexState(
        doc_id="test_doc",
        chunks=chunks,
        concepts=concepts,
        hash_tables=None,
        vector_store=None,
        concept_graph=None,
        ngram_index={"test": {"chunk_0", "chunk_1"}, "concept": {"chunk_0"}},
    )


class TestLRUCache:

    def test_basic_operations(self):
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 3

    def test_eviction_order(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)

        assert cache.has("a")
        assert not cache.has("b")
        assert cache.has("c")

    def test_update_existing(self):
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert cache.size() == 1

    def test_remove(self):
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        assert cache.remove("a")
        assert not cache.has("a")
        assert cache.size() == 0

    def test_remove_nonexistent(self):
        cache = LRUCache(capacity=3)
        assert not cache.remove("missing")

    def test_peek_no_promotion(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.peek("a")
        cache.put("c", 3)
        assert not cache.has("a")

    def test_keys_values_items(self):
        cache = LRUCache(capacity=5)
        cache.put("a", 1)
        cache.put("b", 2)
        assert set(cache.keys()) == {"a", "b"}
        assert set(cache.values()) == {1, 2}
        assert len(cache.items()) == 2

    def test_is_empty_is_full(self):
        cache = LRUCache(capacity=2)
        assert cache.is_empty()
        assert not cache.is_full()

        cache.put("a", 1)
        cache.put("b", 2)
        assert not cache.is_empty()
        assert cache.is_full()

    def test_resize(self):
        cache = LRUCache(capacity=5)
        for i in range(5):
            cache.put(str(i), i)
        cache.resize(3)
        assert cache.size() <= 3
        assert cache.capacity == 3

    def test_get_or_compute(self):
        cache = LRUCache(capacity=5)
        result = cache.get_or_compute("key", lambda k: f"value_for_{k}")
        assert result == "value_for_key"
        result2 = cache.get_or_compute("key", lambda k: "should_not_run")
        assert result2 == "value_for_key"

    def test_update_from_dict(self):
        cache = LRUCache(capacity=5)
        cache.update({"a": 1, "b": 2})
        assert cache.get("a") == 1
        assert cache.get("b") == 2


class TestSerializer:

    def test_serialize_deserialize(self):
        state = _make_test_state()
        serializer = Serializer()

        data = serializer.serialize_index(state)
        assert len(data) > 0

        restored = serializer.deserialize_index(data)
        assert restored.doc_id == state.doc_id
        assert len(restored.chunks) == len(state.chunks)
        assert len(restored.concepts) == len(state.concepts)

    def test_chunks_preserved(self):
        state = _make_test_state()
        serializer = Serializer()

        data = serializer.serialize_index(state)
        restored = serializer.deserialize_index(data)

        for orig, rest in zip(state.chunks, restored.chunks):
            assert rest.chunk_id == orig.chunk_id
            assert rest.text == orig.text
            assert rest.word_count == orig.word_count
            assert rest.page == orig.page
            assert rest.section == orig.section
            assert rest.words == orig.words

    def test_concepts_preserved(self):
        state = _make_test_state()
        serializer = Serializer()

        data = serializer.serialize_index(state)
        restored = serializer.deserialize_index(data)

        for orig, rest in zip(state.concepts, restored.concepts):
            assert rest.concept_id == orig.concept_id
            assert rest.label == orig.label
            assert rest.frequency == orig.frequency
            assert rest.score == orig.score
            assert rest.chunk_ids == orig.chunk_ids

    def test_ngram_index_preserved(self):
        state = _make_test_state()
        serializer = Serializer()

        data = serializer.serialize_index(state)
        restored = serializer.deserialize_index(data)

        assert set(restored.ngram_index.keys()) == set(state.ngram_index.keys())
        for key in state.ngram_index:
            assert restored.ngram_index[key] == state.ngram_index[key]

    def test_empty_state(self):
        state = IndexState(
            doc_id="empty", chunks=[], concepts=[],
            hash_tables=None, vector_store=None,
            concept_graph=None, ngram_index={},
        )
        serializer = Serializer()
        data = serializer.serialize_index(state)
        restored = serializer.deserialize_index(data)
        assert restored.doc_id == "empty"
        assert len(restored.chunks) == 0


class TestFileStore:

    def test_save_and_load(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state)
        assert store.exists("doc1")

        loaded = store.load("doc1")
        assert loaded.doc_id == state.doc_id
        assert len(loaded.chunks) == len(state.chunks)

    def test_list_documents(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state)
        store.save("doc2", state)

        docs = store.list_documents()
        assert "doc1" in docs
        assert "doc2" in docs

    def test_delete(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state)
        assert store.exists("doc1")

        store.delete("doc1")
        assert not store.exists("doc1")

    def test_metadata(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state, {"title": "Test"})
        meta = store.get_metadata("doc1")

        assert meta is not None
        assert meta["doc_id"] == "doc1"
        assert meta["chunk_count"] == 3
        assert meta["title"] == "Test"

    def test_storage_size(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state)
        size = store.get_storage_size()

        assert size["total_bytes"] > 0
        assert size["document_count"] == 1

    def test_nonexistent_load(self, tmp_dir):
        store = FileStore(tmp_dir)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_clear(self, tmp_dir):
        state = _make_test_state()
        store = FileStore(tmp_dir)

        store.save("doc1", state)
        store.save("doc2", state)
        count = store.clear()
        assert count == 2
        assert store.list_documents() == []


class TestProjectStore:

    def test_create_project(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test Project", "A description")
        assert pid

        project = proj.get_project(pid)
        assert project is not None
        assert project["name"] == "Test Project"

    def test_add_document(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test")
        proj.add_document(pid, "doc1", {"title": "Doc 1"})

        docs = proj.list_documents(pid)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc1"

    def test_remove_document(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test")
        proj.add_document(pid, "doc1")
        proj.add_document(pid, "doc2")

        assert proj.remove_document(pid, "doc1")
        docs = proj.list_documents(pid)
        assert len(docs) == 1

    def test_list_projects(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        proj.create_project("Project A")
        proj.create_project("Project B")

        projects = proj.list_projects()
        assert len(projects) == 2

    def test_delete_project(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test")
        assert proj.delete_project(pid)
        assert proj.get_project(pid) is None

    def test_update_project(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Original Name")
        proj.update_project(pid, name="Updated Name")

        project = proj.get_project(pid)
        assert project["name"] == "Updated Name"

    def test_project_stats(self, tmp_dir):
        state = _make_test_state()
        file_store = FileStore(tmp_dir)
        file_store.save("doc1", state)

        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test")
        proj.add_document(pid, "doc1")

        stats = proj.get_project_stats(pid)
        assert stats["document_count"] == 1

    def test_duplicate_document(self, tmp_dir):
        proj = ProjectStore(tmp_dir)
        pid = proj.create_project("Test")
        proj.add_document(pid, "doc1")
        proj.add_document(pid, "doc1")

        docs = proj.list_documents(pid)
        assert len(docs) == 1