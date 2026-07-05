from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..core.types import ParsedDocument


class BaseParser(ABC):

    @abstractmethod
    def parse(self, source: str) -> ParsedDocument:
        pass

    @abstractmethod
    def supports(self, format: str) -> bool:
        pass

    def _make_result(
        self,
        text: str,
        pages: Optional[List[Tuple[int, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        if pages is None:
            pages = [(1, text)] if text else []
        meta = dict(metadata) if metadata else {}
        words = text.split() if text else []
        quality = self._estimate_quality(text)
        return ParsedDocument(
            full_text=text,
            pages=pages,
            metadata=meta,
            page_count=len(pages),
            total_chars=len(text),
            total_words=len(words),
            text_quality=quality,
        )

    def _estimate_quality(self, text: str) -> float:
        if not text:
            return 0.0
        sample = text[:10000]
        alpha = sum(1 for c in sample if c.isalpha() or c.isspace())
        return round(alpha / max(len(sample), 1), 3)

    def _read_text(self, path: str, encoding: Optional[str] = None) -> str:
        if encoding:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        try:
            with open(path, "rb") as f:
                raw = f.read()
            try:
                import chardet

                detected = chardet.detect(raw)
                if detected and detected.get("encoding"):
                    return raw.decode(detected["encoding"], errors="replace")
            except ImportError:
                pass
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _file_exists(self, source: str) -> bool:
        return os.path.isfile(source)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _strip_control_chars(self, text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)