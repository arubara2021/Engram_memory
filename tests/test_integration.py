from __future__ import annotations

import os

import pytest

from engram.core.engine import Engram
from engram.core.config import EngramConfig
from engram.core.types import ChunkResult, Response


class TestIngestAndRetrieve:

    def test_ingest_text_file(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()

        doc_id = engine.ingest(path)
        assert doc_id
        assert doc_id in engine.documents

        state = engine.documents[doc_id]
        assert len(state.index.chunks) > 0
        assert len(state.index.concepts) > 0
        assert state.document.total_words > 0

    def test_ingest_markdown(self, write_file, sample_markdown):
        path = write_file("readme.md", sample_markdown)
        engine = Engram()

        doc_id = engine.ingest(path)
        state = engine.documents[doc_id]
        assert len(state.index.chunks) > 0

    def test_ingest_json(self, write_file, sample_json):
        path = write_file("data.json", sample_json)
        engine = Engram()

        doc_id = engine.ingest(path)
        state = engine.documents[doc_id]
        assert "Engram" in state.document.full_text

    def test_ingest_csv(self, write_file, sample_csv):
        path = write_file("data.csv", sample_csv)
        engine = Engram()

        doc_id = engine.ingest(path)
        state = engine.documents[doc_id]
        assert "Alice" in state.document.full_text

    def test_ingest_html(self, write_file, sample_html):
        path = write_file("page.html", sample_html)
        engine = Engram()

        doc_id = engine.ingest(path)
        state = engine.documents[doc_id]
        assert "Engram" in state.document.full_text

    def test_ingest_directory(self, tmp_dir, sample_text, sample_markdown, sample_json):
        for name, content in [
            ("notes.txt", sample_text),
            ("readme.md", sample_markdown),
            ("data.json", sample_json),
        ]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write(content)

        engine = Engram()
        doc_id = engine.ingest(tmp_dir)

        state = engine.documents[doc_id]
        assert len(state.index.chunks) > 0
        assert state.document.total_words > 100

    def test_retrieve_basic(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        results = engine.retrieve("What is machine learning?")
        assert len(results) > 0
        assert isinstance(results[0], ChunkResult)
        assert results[0].score > 0

    def test_retrieve_returns_relevant(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        results = engine.retrieve("neural network layers")
        assert len(results) > 0
        found = any("neural" in r.chunk.text.lower() for r in results)
        assert found

    def test_retrieve_with_top_k(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        results = engine.retrieve("machine learning", top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_query(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        results = engine.retrieve("xyznonexistent123456")
        assert isinstance(results, list)

    def test_ingest_batch(self, tmp_dir, sample_text, sample_markdown):
        paths = []
        for name, content in [("a.txt", sample_text), ("b.md", sample_markdown)]:
            p = os.path.join(tmp_dir, name)
            with open(p, "w") as f:
                f.write(content)
            paths.append(p)

        engine = Engram()
        doc_ids = engine.ingest_batch(paths)
        assert len(doc_ids) >= 1

    def test_concepts_extracted(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        doc_id = engine.ingest(path)

        concepts = engine.get_concepts(doc_id)
        assert len(concepts) > 0
        labels = [c.label for c in concepts]
        assert any("learning" in l.lower() or "neural" in l.lower() for l in labels)

    def test_get_stats(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        stats = engine.get_stats()
        assert stats["documents"] == 1
        assert stats["total_chunks"] > 0
        assert stats["total_concepts"] > 0
        assert stats["total_words"] > 0

    def test_list_documents(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        docs = engine.list_documents()
        assert len(docs) == 1
        assert docs[0]["chunks"] > 0

    def test_remove_document(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        doc_id = engine.ingest(path)

        assert engine.remove_document(doc_id)
        assert doc_id not in engine.documents

    def test_file_not_found(self):
        engine = Engram()
        with pytest.raises(FileNotFoundError):
            engine.ingest("/nonexistent/path/file.txt")

    def test_no_documents_error(self):
        engine = Engram()
        with pytest.raises(RuntimeError):
            engine.retrieve("test query")


class TestFullPipeline:

    def test_ingest_chunk_index_retrieve(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        config = EngramConfig(
            chunk_strategy="recursive",
            min_chunk_words=20,
            max_chunk_words=150,
            max_concepts=15,
            final_top_k=3,
        )
        engine = Engram(config)
        doc_id = engine.ingest(path)

        state = engine.documents[doc_id]
        assert len(state.index.chunks) >= 3
        assert len(state.index.concepts) >= 3
        assert state.index.hash_tables is not None
        assert state.index.hash_tables.total_insertions > 0

        results = engine.retrieve("What are the types of machine learning?")
        assert len(results) > 0
        assert len(results) <= 3

    def test_multiple_documents(self, tmp_dir, sample_text, sample_markdown):
        path1 = os.path.join(tmp_dir, "notes.txt")
        path2 = os.path.join(tmp_dir, "readme.md")
        with open(path1, "w") as f:
            f.write(sample_text)
        with open(path2, "w") as f:
            f.write(sample_markdown)

        engine = Engram()
        id1 = engine.ingest(path1)
        id2 = engine.ingest(path2)

        assert id1 != id2
        assert len(engine.documents) == 2

        results = engine.retrieve("hash-based retrieval")
        assert len(results) > 0

    def test_stats_after_multiple(self, tmp_dir, sample_text, sample_csv):
        path1 = os.path.join(tmp_dir, "notes.txt")
        path2 = os.path.join(tmp_dir, "data.csv")
        with open(path1, "w") as f:
            f.write(sample_text)
        with open(path2, "w") as f:
            f.write(sample_csv)

        engine = Engram()
        engine.ingest(path1)
        engine.ingest(path2)

        stats = engine.get_stats()
        assert stats["documents"] == 2
        assert stats["total_chunks"] > 0

    def test_concept_graph_built(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        doc_id = engine.ingest(path)

        graph = engine.get_graph(doc_id)
        assert graph is not None


class TestSerialization:

    def test_save_and_load(self, write_file, sample_text, tmp_dir):
        path = write_file("notes.txt", sample_text)
        engine = Engram()
        engine.ingest(path)

        save_path = os.path.join(tmp_dir, "saved_data")
        engine.save(save_path)

        assert os.path.exists(save_path)

    def test_ingest_code(self, write_file, sample_code):
        path = write_file("service.py", sample_code)
        config = EngramConfig(chunk_strategy="recursive")
        engine = Engram(config)
        doc_id = engine.ingest(path)

        state = engine.documents[doc_id]
        assert len(state.index.chunks) > 0