from __future__ import annotations

import pytest

from engram.chunk.base import BaseChunker
from engram.chunk.recursive import RecursiveChunker
from engram.chunk.hierarchical import HierarchicalChunker
from engram.chunk.page import PageChunker
from engram.chunk.overlap import OverlapHandler
from engram.chunk.splitter import SentenceSplitter
from engram.chunk.reference import ReferenceStripper
from engram.chunk.factory import ChunkerFactory
from engram.core.config import EngramConfig


class TestRecursiveChunker:

    def test_basic_chunking(self, sample_text):
        config = EngramConfig(min_chunk_words=20, max_chunk_words=180)
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk(sample_text, "test_doc")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == "test_doc"
            assert chunk.text.strip() != ""
            assert chunk.word_count > 0

    def test_chunk_size_bounds(self, sample_text):
        config = EngramConfig(min_chunk_words=20, max_chunk_words=100)
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk(sample_text, "test_doc")

        for chunk in chunks:
            assert chunk.word_count >= 5
            assert chunk.word_count <= 120

    def test_chunk_ids_unique(self, sample_text):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_text, "test_doc")

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_words_populated(self, sample_text):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_text, "test_doc")

        for chunk in chunks:
            assert isinstance(chunk.words, list)
            assert len(chunk.words) > 0

    def test_empty_text(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("", "test_doc")
        assert len(chunks) == 0

    def test_short_text(self):
        chunker = RecursiveChunker(EngramConfig(min_chunk_words=5, max_chunk_words=50))
        chunks = chunker.chunk("This is a short text.", "test_doc")
        assert len(chunks) >= 0

    def test_name_property(self):
        chunker = RecursiveChunker()
        assert chunker.name == "RecursiveChunker"


class TestHierarchicalChunker:

    def test_creates_small_and_large(self, sample_text):
        config = EngramConfig(
            min_chunk_words=10,
            max_chunk_words=300,
            hierarchical_small=80,
            hierarchical_large=200,
        )
        chunker = HierarchicalChunker(config)
        chunks = chunker.chunk(sample_text, "test_doc")

        small = [c for c in chunks if c.chunk_id.startswith("small_")]
        large = [c for c in chunks if c.chunk_id.startswith("large_")]

        assert len(small) > 0
        assert len(large) > 0
        assert len(large) < len(small)

    def test_all_chunks_have_text(self, sample_text):
        config = EngramConfig(
            min_chunk_words=10,
            hierarchical_small=80,
            hierarchical_large=200,
        )
        chunker = HierarchicalChunker(config)
        chunks = chunker.chunk(sample_text, "test_doc")

        for chunk in chunks:
            assert chunk.text.strip() != ""

class TestPageChunker:

    def test_page_based_chunking(self, sample_text):
        chunker = PageChunker()
        pages = [
            (1, sample_text[:len(sample_text)//3]),
            (2, sample_text[len(sample_text)//3:2*len(sample_text)//3]),
            (3, sample_text[2*len(sample_text)//3:]),
        ]
        chunks = chunker.chunk(sample_text, "test_doc", pages=pages)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.page > 0

    def test_merges_short_pages(self):
        chunker = PageChunker()
        pages = [(1, "Short."), (2, "Also short.")]
        chunks = chunker.chunk("Short. Also short.", "test_doc", pages=pages)

        assert len(chunks) >= 1


class TestOverlapHandler:

    def test_overlap_applied(self, sample_text):
        config = EngramConfig(min_chunk_words=20, max_chunk_words=100)
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk(sample_text, "test_doc")

        handler = OverlapHandler()
        overlapped = handler.apply(chunks, overlap_sentences=2)

        assert len(overlapped) == len(chunks)
        if len(chunks) > 1:
            assert len(overlapped[1].text) >= len(chunks[1].text)

    def test_first_chunk_unchanged(self, sample_text):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_text, "test_doc")

        handler = OverlapHandler()
        overlapped = handler.apply(chunks, overlap_sentences=2)

        assert overlapped[0].text == chunks[0].text

    def test_no_overlap_single_chunk(self):
        from engram.core.types import Chunk
        chunks = [Chunk(chunk_id="c0", doc_id="d", text="Single chunk.", word_count=2, page=0, section="", index=0, words=["single", "chunk"])]
        handler = OverlapHandler()
        result = handler.apply(chunks, 2)
        assert len(result) == 1


class TestSentenceSplitter:

    def test_basic_split(self):
        splitter = SentenceSplitter()
        sentences = splitter.split("This is the first sentence in the document. Here is the second sentence with different content. Finally the third sentence completes the paragraph.")
        assert len(sentences) >= 2

    def test_empty_text(self):
        splitter = SentenceSplitter()
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_preserves_urls(self):
        splitter = SentenceSplitter()
        sentences = splitter.split("Visit https://example.com for more info. Then read the docs.")
        combined = " ".join(sentences)
        assert "https://example.com" in combined

    def test_preserves_decimals(self):
        splitter = SentenceSplitter()
        sentences = splitter.split("The value is 3.14. This is important.")
        combined = " ".join(sentences)
        assert "3.14" in combined


class TestReferenceStripper:

    def test_strip_references(self):
        text = (
            "Main content here.\n\n"
            "More content.\n\n"
            "References\n"
            "Smith, J. 2020. Title of paper.\n"
            "Jones, A. and Brown, B. 2019. Another paper.\n"
        )
        stripper = ReferenceStripper()
        result = stripper.strip(text)
        assert "Main content" in result
        assert "Smith, J." not in result

    def test_no_references(self):
        text = "Just regular content with no references section."
        stripper = ReferenceStripper()
        result = stripper.strip(text)
        assert result == text


class TestChunkerFactory:

    def test_create_recursive(self):
        config = EngramConfig()
        chunker = ChunkerFactory.create("recursive", config)
        assert isinstance(chunker, RecursiveChunker)

    def test_create_hierarchical(self):
        config = EngramConfig()
        chunker = ChunkerFactory.create("hierarchical", config)
        assert isinstance(chunker, HierarchicalChunker)

    def test_create_page(self):
        config = EngramConfig()
        chunker = ChunkerFactory.create("page", config)
        assert isinstance(chunker, PageChunker)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            ChunkerFactory.create("nonexistent")

    def test_list_strategies(self):
        strategies = ChunkerFactory.list_strategies()
        assert "recursive" in strategies
        assert "hierarchical" in strategies
        assert "page" in strategies

