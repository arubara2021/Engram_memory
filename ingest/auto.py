from __future__ import annotations

import os
import re
from typing import Dict, Optional


EXTENSION_MAP: Dict[str, str] = {
    "pdf": "pdf",
    "txt": "text",
    "text": "text",
    "log": "text",
    "eml": "text",
    "html": "html",
    "htm": "html",
    "xhtml": "html",
    "md": "markdown",
    "markdown": "markdown",
    "mkd": "markdown",
    "mdown": "markdown",
    "rst": "rst",
    "rest": "rst",
    "xml": "xml",
    "svg": "xml",
    "docx": "docx",
    "doc": "doc",
    "rtf": "rtf",
    "odt": "odt",
    "pptx": "pptx",
    "ppt": "ppt",
    "odp": "odp",
    "xlsx": "xlsx",
    "xls": "xls",
    "ods": "ods",
    "csv": "csv",
    "tsv": "csv",
    "json": "json",
    "jsonl": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "py": "code",
    "pyw": "code",
    "js": "code",
    "mjs": "code",
    "cjs": "code",
    "ts": "code",
    "tsx": "code",
    "jsx": "code",
    "java": "code",
    "c": "code",
    "cpp": "code",
    "cc": "code",
    "cxx": "code",
    "h": "code",
    "hpp": "code",
    "hxx": "code",
    "cs": "code",
    "go": "code",
    "rs": "code",
    "rb": "code",
    "php": "code",
    "swift": "code",
    "kt": "code",
    "kts": "code",
    "scala": "code",
    "r": "code",
    "m": "code",
    "sh": "code",
    "bash": "code",
    "zsh": "code",
    "fish": "code",
    "ps1": "code",
    "bat": "code",
    "cmd": "code",
    "sql": "code",
    "css": "code",
    "scss": "code",
    "sass": "code",
    "less": "code",
    "lua": "code",
    "pl": "code",
    "pm": "code",
    "ex": "code",
    "exs": "code",
    "erl": "code",
    "hs": "code",
    "lhs": "code",
    "ml": "code",
    "mli": "code",
    "clj": "code",
    "cljs": "code",
    "cljc": "code",
    "lisp": "code",
    "el": "code",
    "rkt": "code",
    "dart": "code",
    "zig": "code",
    "nim": "code",
    "v": "code",
    "jl": "code",
    "vue": "code",
    "svelte": "code",
    "dockerfile": "code",
    "makefile": "code",
    "cmake": "code",
    "gradle": "code",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "bmp": "image",
    "tiff": "image",
    "tif": "image",
    "webp": "image",
    "ico": "image",
    "svg": "image",
}

CONTENT_TYPE_MAP: Dict[str, str] = {
    "application/pdf": "pdf",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/plain": "text",
    "text/csv": "csv",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/msword": "doc",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.ms-excel": "xls",
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/tiff": "image",
}


class FormatDetector:

    def detect(self, source: str) -> str:
        ext = os.path.splitext(source)[1].lower().lstrip(".")
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]
        return self._detect_from_content(source)

    def detect_from_extension(self, path: str) -> Optional[str]:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        return EXTENSION_MAP.get(ext)

    def detect_from_content_type(self, content_type: str) -> Optional[str]:
        ct = content_type.split(";")[0].strip().lower()
        return CONTENT_TYPE_MAP.get(ct)

    def detect_from_bytes(self, data: bytes) -> str:
        if len(data) < 4:
            return "text"
        if data[:5] == b"%PDF-":
            return "pdf"
        if data[:2] == b"PK":
            return self._detect_zip_content(data)
        if data[:3] == b"\xef\xbb\xbf":
            return "text"
        try:
            text_start = data[:200].decode("utf-8", errors="ignore").strip()
            if text_start.startswith(("{", "[")):
                return "json"
            if text_start.startswith("---"):
                return "yaml"
            if text_start.startswith("<?xml") or text_start.startswith("<"):
                if b"<html" in data[:1000].lower() or b"<!doctype" in data[:1000].lower():
                    return "html"
                return "xml"
        except Exception:
            pass
        printable = sum(1 for b in data[:1000] if 32 <= b <= 126 or b in (9, 10, 13))
        if printable / min(len(data), 1000) > 0.7:
            return "text"
        return "binary"

    def _detect_from_content(self, source: str) -> str:
        if not os.path.exists(source):
            return "text"
        try:
            with open(source, "rb") as f:
                header = f.read(4096)
            return self.detect_from_bytes(header)
        except Exception:
            return "text"

    def _detect_zip_content(self, data: bytes) -> str:
        import io
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                for name in names:
                    lower = name.lower()
                    if "word/" in lower:
                        return "docx"
                    if "ppt/" in lower:
                        return "pptx"
                    if "xl/" in lower:
                        return "xlsx"
                if any(n.lower().endswith(".opf") for n in names):
                    return "epub"
        except Exception:
            pass
        return "zip"

    def get_extension_map(self) -> Dict[str, str]:
        return dict(EXTENSION_MAP)

    def list_supported(self) -> list:
        return sorted(set(EXTENSION_MAP.values()))