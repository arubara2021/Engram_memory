from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


class PDFParser(BaseParser):

    def __init__(self) -> None:
        pass

    def supports(self, format: str) -> bool:
        return format.lower() in ("pdf",)

    def parse(self, source: str) -> ParsedDocument:
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. "
                "Install it with: pip install pymupdf"
            )

        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        doc = fitz.open(source)
        page_count = len(doc)

        pages: List[Tuple[int, str]] = []
        page_spans: List[Tuple[int, int, int]] = []

        running_text = ""
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                start = len(running_text)
                if running_text:
                    running_text += "\n\n"
                running_text += text.strip()
                end = len(running_text)
                page_spans.append((start, end, page_num + 1))
                pages.append((page_num + 1, text.strip()))

        doc.close()

        full_text = running_text
        words = full_text.split() if full_text else []
        quality = self._estimate_quality(full_text)

        metadata: Dict[str, Any] = {
            "format": "pdf",
            "page_count": page_count,
            "title": self._extract_title(full_text, source),
        }

        return ParsedDocument(
            full_text=full_text,
            pages=pages,
            metadata=metadata,
            page_count=page_count,
            total_chars=len(full_text),
            total_words=len(words),
            text_quality=quality,
            page_spans=page_spans,
        )

    def _extract_title(self, text: str, filepath: str) -> str:
        if text:
            lines = text.strip().split("\n")
            for line in lines[:10]:
                line = line.strip()
                if line and 5 < len(line) < 100:
                    return line
        filename = os.path.basename(filepath)
        return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
