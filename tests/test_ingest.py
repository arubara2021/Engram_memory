from __future__ import annotations

import os

import pytest

from engram.ingest.auto import FormatDetector
from engram.ingest.base import BaseParser
from engram.ingest.cleaner import TextCleaner
from engram.ingest.metadata import MetadataDetector
from engram.ingest.data import DataParser
from engram.ingest.markup import MarkupParser
from engram.ingest.text import TextParser


class TestFormatDetector:

    def test_detect_pdf(self, write_file):
        path = write_file("test.pdf", "not really a pdf")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "pdf"

    def test_detect_text(self, write_file):
        path = write_file("notes.txt", "hello world")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "text"

    def test_detect_json(self, write_file):
        path = write_file("data.json", '{"key": "value"}')
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "json"

    def test_detect_markdown(self, write_file):
        path = write_file("readme.md", "# Hello")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "markdown"

    def test_detect_html(self, write_file):
        path = write_file("page.html", "<html><body>hi</body></html>")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "html"

    def test_detect_csv(self, write_file):
        path = write_file("data.csv", "a,b,c\n1,2,3")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "csv"

    def test_detect_yaml(self, write_file):
        path = write_file("config.yaml", "key: value")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "yaml"

    def test_detect_python(self, write_file):
        path = write_file("main.py", "print('hello')")
        detector = FormatDetector()
        result = detector.detect(path)
        assert result == "code"

    def test_detect_from_bytes_pdf(self):
        detector = FormatDetector()
        result = detector.detect_from_bytes(b"%PDF-1.4 some content")
        assert result == "pdf"

    def test_detect_from_bytes_json(self):
        detector = FormatDetector()
        result = detector.detect_from_bytes(b'{"key": "value"}')
        assert result == "json"

    def test_detect_from_bytes_yaml(self):
        detector = FormatDetector()
        result = detector.detect_from_bytes(b"---\nkey: value\n")
        assert result == "yaml"

    def test_detect_from_bytes_html(self):
        detector = FormatDetector()
        result = detector.detect_from_bytes(b"<html><body>test</body></html>")
        assert result == "html"

    def test_detect_from_bytes_text(self):
        detector = FormatDetector()
        result = detector.detect_from_bytes(b"just plain text here")
        assert result == "text"

    def test_content_type_mapping(self):
        detector = FormatDetector()
        assert detector.detect_from_content_type("application/pdf") == "pdf"
        assert detector.detect_from_content_type("text/html") == "html"
        assert detector.detect_from_content_type("application/json") == "json"
        assert detector.detect_from_content_type("text/csv") == "csv"

    def test_list_supported(self):
        detector = FormatDetector()
        supported = detector.list_supported()
        assert "pdf" in supported
        assert "json" in supported
        assert "html" in supported
        assert "code" in supported
        assert len(supported) > 5


class TestTextCleaner:

    def test_basic_clean(self):
        cleaner = TextCleaner()
        result = cleaner.clean("  hello   world  ")
        assert result == "hello world"

    def test_ligature_fix(self):
        cleaner = TextCleaner()
        result = cleaner.clean("\ufb01nance")
        assert "fi" in result

    def test_smart_quotes(self):
        cleaner = TextCleaner()
        result = cleaner.clean("\u201chello\u201d \u2018world\u2019")
        assert '"' in result
        assert "'" in result

    def test_hyphenation_fix(self):
        cleaner = TextCleaner()
        result = cleaner.clean("informa-\ntion")
        assert "information" in result

    def test_whitespace_normalization(self):
        cleaner = TextCleaner()
        result = cleaner.clean("a\t\tb\n\n\n\nc")
        assert "\t" not in result
        assert "\n\n\n" not in result

    def test_control_char_removal(self):
        cleaner = TextCleaner()
        result = cleaner.clean("hello\x00\x01world")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result

    def test_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""


class TestMetadataDetector:

    def test_detect_title_from_heading(self):
        detector = MetadataDetector()
        text = "# My Document Title\n\nSome content here."
        meta = detector.detect(text)
        assert meta.get("title") == "My Document Title"

    def test_detect_title_from_filename(self):
        detector = MetadataDetector()
        meta = detector.detect("Some content", "My_Report.pdf")
        assert meta.get("title") == "My Report"

    def test_detect_language_english(self):
        detector = MetadataDetector()
        text = "The system is designed to process documents and extract useful information."
        meta = detector.detect(text)
        assert meta.get("language") == "en"

    def test_detect_domain_academic(self):
        detector = MetadataDetector()
        text = (
            "Abstract: This paper presents a novel methodology for analyzing "
            "experimental results. The hypothesis was tested through a controlled "
            "experiment. References cited in the bibliography support our findings. "
            "The literature review covers recent proceedings from peer reviewed journals."
        )
        meta = detector.detect(text)
        assert meta.get("domain") == "academic"

    def test_detect_domain_technical(self):
        detector = MetadataDetector()
        text = (
            "The algorithm implementation uses a framework with modular architecture. "
            "Each component has a well-defined interface. The configuration file "
            "controls deployment to the server. Debug mode helps with optimization."
        )
        meta = detector.detect(text)
        assert meta.get("domain") == "technical"

    def test_detect_headings(self):
        detector = MetadataDetector()
        text = (
            "# Main Title\n"
            "## Section One\n"
            "Content here.\n"
            "## Section Two\n"
            "More content.\n"
            "### Subsection\n"
            "Details here."
        )
        meta = detector.detect(text)
        assert meta.get("heading_count", 0) >= 3

    def test_structure_analysis(self):
        detector = MetadataDetector()
        text = "Line one.\n\nLine two.\n\nLine three."
        meta = detector.detect(text)
        assert meta.get("line_count", 0) > 0
        assert meta.get("word_count", 0) > 0


class TestTextParser:

    def test_parse_text_file(self, write_file, sample_text):
        path = write_file("notes.txt", sample_text)
        parser = TextParser()
        result = parser.parse(path)
        assert result.total_chars > 0
        assert result.total_words > 0
        assert result.text_quality > 0
        assert "Machine learning" in result.full_text

    def test_parse_log_file(self, write_file):
        content = (
            "2024-01-15 10:30:00 INFO Application started\n"
            "2024-01-15 10:30:05 WARNING Low memory\n"
            "2024-01-15 10:31:00 ERROR Connection failed\n"
        )
        path = write_file("app.log", content)
        parser = TextParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "log"
        assert result.metadata.get("entry_count", 0) >= 3

    def test_supports(self):
        parser = TextParser()
        assert parser.supports("text")
        assert parser.supports("txt")
        assert parser.supports("log")
        assert parser.supports("eml")


class TestDataParser:

    def test_parse_json(self, write_file, sample_json):
        path = write_file("data.json", sample_json)
        parser = DataParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "json"
        assert "Engram Project" in result.full_text
        assert "hash-based retrieval" in result.full_text.lower()

    def test_parse_csv(self, write_file, sample_csv):
        path = write_file("people.csv", sample_csv)
        parser = DataParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "csv"
        assert "Alice" in result.full_text
        assert result.metadata.get("row_count", 0) == 5

    def test_parse_yaml(self, write_file, sample_yaml):
        path = write_file("config.yaml", sample_yaml)
        parser = DataParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "yaml"
        assert "engram" in result.full_text.lower()

    def test_parse_toml(self, write_file, sample_toml):
        path = write_file("config.toml", sample_toml)
        parser = DataParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "toml"
        assert "engram" in result.full_text.lower()

    def test_parse_jsonl(self, write_file):
        content = '{"name": "Alice", "age": 30}\n{"name": "Bob", "age": 25}\n'
        path = write_file("data.jsonl", content)
        parser = DataParser()
        result = parser.parse(path)
        assert "Alice" in result.full_text
        assert "Bob" in result.full_text

    def test_supports(self):
        parser = DataParser()
        assert parser.supports("json")
        assert parser.supports("csv")
        assert parser.supports("yaml")
        assert parser.supports("toml")


class TestMarkupParser:

    def test_parse_markdown(self, write_file, sample_markdown):
        path = write_file("readme.md", sample_markdown)
        parser = MarkupParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "markdown"
        assert "Engram" in result.full_text
        assert "hash-based retrieval" in result.full_text.lower()

    def test_parse_html(self, write_file, sample_html):
        path = write_file("page.html", sample_html)
        parser = MarkupParser()
        result = parser.parse(path)
        assert result.metadata.get("format") == "html"
        assert "Engram" in result.full_text
        assert "console.log" not in result.full_text
        assert "font-family" not in result.full_text

    def test_markdown_heading_extraction(self, write_file, sample_markdown):
        path = write_file("doc.md", sample_markdown)
        parser = MarkupParser()
        result = parser.parse(path)
        assert result.metadata.get("title") == "Engram Documentation"

    def test_markdown_code_blocks_preserved(self, write_file, sample_markdown):
        path = write_file("doc.md", sample_markdown)
        parser = MarkupParser()
        result = parser.parse(path)
        assert "pip install engram" in result.full_text

    def test_supports(self):
        parser = MarkupParser()
        assert parser.supports("html")
        assert parser.supports("md")
        assert parser.supports("rst")


class TestPDFParser:

    def test_parse_pdf(self, write_file):
        pytest.importorskip("fitz")
        from engram.ingest.pdf import PDFParser

        parser = PDFParser()
        assert parser.supports("pdf")

    def test_missing_file(self, tmp_dir):
        pytest.importorskip("fitz")
        from engram.ingest.pdf import PDFParser

        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(os.path.join(tmp_dir, "nonexistent.pdf"))