from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    "py": "python", "pyw": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript", "jsx": "javascript",
    "java": "java",
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hxx": "cpp",
    "cs": "csharp",
    "go": "go",
    "rs": "rust",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin", "kts": "kotlin",
    "scala": "scala",
    "r": "r", "R": "r",
    "m": "objc",
    "sh": "shell", "bash": "shell", "zsh": "shell", "fish": "shell",
    "ps1": "powershell", "bat": "batch", "cmd": "batch",
    "sql": "sql",
    "css": "css", "scss": "scss", "sass": "sass", "less": "less",
    "lua": "lua",
    "pl": "perl", "pm": "perl",
    "ex": "elixir", "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell", "lhs": "haskell",
    "ml": "ocaml", "mli": "ocaml",
    "clj": "clojure", "cljs": "clojure", "cljc": "clojure",
    "lisp": "lisp", "el": "emacs_lisp",
    "rkt": "racket",
    "dart": "dart",
    "zig": "zig",
    "nim": "nim",
    "v": "v",
    "jl": "julia",
    "vue": "vue",
    "svelte": "svelte",
    "dockerfile": "docker",
    "makefile": "make",
    "cmake": "cmake",
    "gradle": "gradle",
}

COMMENT_STYLES: Dict[str, Dict[str, Any]] = {
    "python": {"single": "#", "multi_start": '"""', "multi_end": '"""', "docstring": True},
    "javascript": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "typescript": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "java": {"single": "//", "multi_start": "/*", "multi_end": "*/", "doc": True},
    "c": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "cpp": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "csharp": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "go": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "rust": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "ruby": {"single": "#", "multi_start": "=begin", "multi_end": "=end"},
    "php": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "swift": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "kotlin": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "scala": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "shell": {"single": "#"},
    "sql": {"single": "--", "multi_start": "/*", "multi_end": "*/"},
    "lua": {"single": "--", "multi_start": "--[[", "multi_end": "]]"},
    "perl": {"single": "#"},
    "haskell": {"single": "--", "multi_start": "{-", "multi_end": "-}"},
    "r": {"single": "#"},
    "elixir": {"single": "#"},
    "erlang": {"single": "%"},
    "clojure": {"single": ";"},
    "lisp": {"single": ";"},
    "emacs_lisp": {"single": ";"},
    "racket": {"single": ";"},
    "dart": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
    "zig": {"single": "//"},
    "nim": {"single": "#"},
    "julia": {"single": "#"},
    "css": {"multi_start": "/*", "multi_end": "*/"},
    "scss": {"single": "//", "multi_start": "/*", "multi_end": "*/"},
}

PYTHON_PATTERNS = [
    (re.compile(r"^class\s+(\w+)(?:$$[^)]*$$)?\s*:", re.MULTILINE), "class"),
    (re.compile(r"^(async\s+)?def\s+(\w+)\s*$$[^)]*$$\s*(?:->[^:]*)?:", re.MULTILINE), "function"),
]

GENERAL_PATTERNS = [
    (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*$$", re.MULTILINE), "function"),
    (re.compile(r"^(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", re.MULTILINE), "function"),
    (re.compile(r"^(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function", re.MULTILINE), "function"),
    (re.compile(r"^class\s+(\w+)", re.MULTILINE), "class"),
    (re.compile(r"^interface\s+(\w+)", re.MULTILINE), "interface"),
    (re.compile(r"^type\s+(\w+)", re.MULTILINE), "type"),
    (re.compile(r"^enum\s+(\w+)", re.MULTILINE), "enum"),
    (re.compile(r"^struct\s+(\w+)", re.MULTILINE), "struct"),
    (re.compile(r"^(?:public|private|protected|static|async|virtual|override|abstract)\s+.*?\s+(\w+)\s*\(", re.MULTILINE), "method"),
]

IMPORT_PATTERNS: Dict[str, re.Pattern] = {
    "python": re.compile(r"^(?:from\s+\S+\s+)?import\s+.+", re.MULTILINE),
    "javascript": re.compile(r"^import\s+.+from\s+['\"]|^(?:const|let|var)\s+\w+\s*=\s*require\(", re.MULTILINE),
    "typescript": re.compile(r"^import\s+.+from\s+['\"]|^(?:const|let|var)\s+\w+\s*=\s*require\(", re.MULTILINE),
    "java": re.compile(r"^import\s+[\w.]+;?$", re.MULTILINE),
    "go": re.compile(r'^import\s+(?:"[^"]+"|\()', re.MULTILINE),
    "rust": re.compile(r"^use\s+[\w:]+;?$", re.MULTILINE),
    "c": re.compile(r"^#\s*include\s+[<\"].+[>\"]$", re.MULTILINE),
    "cpp": re.compile(r"^#\s*include\s+[<\"].+[>\"]$|^using\s+namespace\s+", re.MULTILINE),
    "ruby": re.compile(r"^(?:require|require_relative)\s+['\"]", re.MULTILINE),
    "php": re.compile(r"^use\s+[\w\\]+;?$|^require(?:_once)?\s", re.MULTILINE),
    "swift": re.compile(r"^import\s+\w+$", re.MULTILINE),
    "kotlin": re.compile(r"^import\s+[\w.]+$", re.MULTILINE),
    "scala": re.compile(r"^import\s+[\w._]+$", re.MULTILINE),
    "shell": re.compile(r"^(?:source|\.)\s+", re.MULTILINE),
}


class CodeParser(BaseParser):

    def supports(self, format: str) -> bool:
        return format.lower() in ("code",) or format.lower() in EXTENSION_TO_LANGUAGE

    def parse(self, source: str) -> ParsedDocument:
        if not self._file_exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        raw_text = self._read_text(source)
        ext = os.path.splitext(source)[1].lower().lstrip(".")
        language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

        if ext in ("dockerfile", "makefile", "cmake", "gradle"):
            language = ext

        structure = self._extract_structure(raw_text, language)
        readable = self._to_readable_text(raw_text, structure, language, source)
        readable = self._normalize_whitespace(readable)

        metadata = {
            "format": "code",
            "language": language,
            "file_name": os.path.basename(source),
            "line_count": raw_text.count("\n") + 1,
            "function_count": len(structure.get("functions", [])),
            "class_count": len(structure.get("classes", [])),
            "import_count": len(structure.get("imports", [])),
            "has_docstrings": structure.get("has_docstrings", False),
        }

        pages = [(1, readable)]
        return self._make_result(readable, pages, metadata)

    def _extract_structure(self, code: str, language: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "imports": [],
            "functions": [],
            "classes": [],
            "interfaces": [],
            "types": [],
            "enums": [],
            "constants": [],
            "comments": [],
            "has_docstrings": False,
        }

        comment_style = COMMENT_STYLES.get(language, {})
        result["comments"] = self._extract_comments(code, comment_style)
        result["has_docstrings"] = any(
            c["is_docstring"] for c in result["comments"]
        )

        import_pattern = IMPORT_PATTERNS.get(language)
        if import_pattern:
            result["imports"] = import_pattern.findall(code)

        if language == "python":
            result["functions"] = self._extract_python_functions(code)
            result["classes"] = self._extract_python_classes(code)
        else:
            result["functions"] = self._extract_general_functions(code)
            result["classes"] = self._extract_general_classes(code)
            result["interfaces"] = self._extract_general_interfaces(code)
            result["types"] = self._extract_general_types(code)
            result["enums"] = self._extract_general_enums(code)

        return result

    def _extract_comments(self, code: str, style: Dict[str, Any]) -> List[Dict[str, Any]]:
        comments: List[str] = []

        single = style.get("single")
        if single:
            pattern = re.compile(
                rf"^{re.escape(single)}\s*(.+)$", re.MULTILINE
            )
            comments.extend(pattern.findall(code))

        multi_start = style.get("multi_start")
        multi_end = style.get("multi_end")
        if multi_start and multi_end:
            pattern = re.compile(
                rf"{re.escape(multi_start)}(.*?){re.escape(multi_end)}",
                re.DOTALL,
            )
            comments.extend(pattern.findall(code))

        result: List[Dict[str, Any]] = []
        seen: set = set()
        for comment in comments:
            text = comment.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            is_doc = False
            if style.get("docstring") or style.get("doc"):
                if len(text) > 20:
                    is_doc = True
            result.append({"text": text, "is_docstring": is_doc})

        return result

    def _extract_python_functions(self, code: str) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        pattern = re.compile(
            r"^((?:async\s+)?def\s+(\w+)\s*\([^)]*\)$$(?:\s*->\s*[^:]+)?)\s*:",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            name = match.group(2)
            signature = match.group(1).strip()
            start = match.end()
            docstring = self._extract_python_docstring(code, start)
            line_num = code[:match.start()].count("\n") + 1
            functions.append({
                "name": name,
                "signature": signature,
                "line": line_num,
                "docstring": docstring,
            })
        return functions

    def _extract_python_docstring(self, code: str, after_colon: int) -> str:
        rest = code[after_colon:].lstrip()
        for marker in ('"""', "'''"):
            if rest.startswith(marker):
                end = rest.find(marker, 3)
                if end != -1:
                    return rest[3:end].strip()
        return ""

    def _extract_python_classes(self, code: str) -> List[Dict[str, Any]]:
        classes: List[Dict[str, Any]] = []
        pattern = re.compile(
            r"^(class\s+(\w+)(?:$$[^)]*$$)?\s*:)", re.MULTILINE
        )
        for match in pattern.finditer(code):
            name = match.group(2)
            signature = match.group(1).strip()
            start = match.end()
            docstring = self._extract_python_docstring(code, start)
            line_num = code[:match.start()].count("\n") + 1
            methods = self._extract_class_methods(code, start, "python")
            classes.append({
                "name": name,
                "signature": signature,
                "line": line_num,
                "docstring": docstring,
                "methods": methods,
            })
        return classes

    def _extract_class_methods(
        self, code: str, start: int, language: str
    ) -> List[str]:
        methods: List[str] = []
        lines = code[start:].split("\n")
        base_indent = None

        for line in lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if base_indent is None:
                if line.strip():
                    base_indent = indent
                continue
            if indent <= base_indent and line.strip() and not line.startswith(" "):
                break
            if indent > (base_indent or 0):
                match = re.match(r"\s+(?:async\s+)?def\s+(\w+)", line)
                if match:
                    methods.append(match.group(1))
        return methods

    def _extract_general_functions(self, code: str) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern, kind in GENERAL_PATTERNS:
            if kind != "function":
                continue
            for match in pattern.finditer(code):
                name = None
                for g in match.groups():
                    if g and re.match(r"^[a-zA-Z_]\w*$", g):
                        name = g
                        break
                if name and name not in seen:
                    seen.add(name)
                    line_num = code[:match.start()].count("\n") + 1
                    functions.append({
                        "name": name,
                        "signature": match.group(0).strip(),
                        "line": line_num,
                        "docstring": "",
                    })
        return functions

    def _extract_general_classes(self, code: str) -> List[Dict[str, Any]]:
        classes: List[Dict[str, Any]] = []
        seen: set = set()
        pattern = re.compile(
            r"^(?:export\s+)?class\s+(\w+)", re.MULTILINE
        )
        for match in pattern.finditer(code):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                line_num = code[:match.start()].count("\n") + 1
                classes.append({
                    "name": name,
                    "signature": match.group(0).strip(),
                    "line": line_num,
                    "docstring": "",
                    "methods": [],
                })
        return classes

    def _extract_general_interfaces(self, code: str) -> List[str]:
        return [
            m.group(1)
            for m in re.finditer(
                r"^(?:export\s+)?interface\s+(\w+)", code, re.MULTILINE
            )
        ]

    def _extract_general_types(self, code: str) -> List[str]:
        return [
            m.group(1)
            for m in re.finditer(
                r"^(?:export\s+)?type\s+(\w+)", code, re.MULTILINE
            )
        ]

    def _extract_general_enums(self, code: str) -> List[str]:
        return [
            m.group(1)
            for m in re.finditer(
                r"^(?:export\s+)?enum\s+(\w+)", code, re.MULTILINE
            )
        ]

    def _to_readable_text(
        self,
        code: str,
        structure: Dict[str, Any],
        language: str,
        source: str,
    ) -> str:
        parts: List[str] = []
        file_name = os.path.basename(source)

        parts.append(f"File: {file_name}")
        if language != "unknown":
            parts.append(f"Language: {language}")
        parts.append("")

        if structure["imports"]:
            parts.append(f"Imports ({len(structure['imports'])}):")
            for imp in structure["imports"][:20]:
                parts.append(f"  {imp.strip()}")
            if len(structure["imports"]) > 20:
                parts.append(f"  ... and {len(structure['imports']) - 20} more")
            parts.append("")

        if structure["classes"]:
            parts.append(f"Classes ({len(structure['classes'])}):")
            for cls in structure["classes"]:
                parts.append(f"  {cls['signature']}")
                if cls.get("docstring"):
                    doc_preview = cls["docstring"][:200].replace("\n", " ")
                    parts.append(f"    Doc: {doc_preview}")
                if cls.get("methods"):
                    parts.append(f"    Methods: {', '.join(cls['methods'][:10])}")
            parts.append("")

        if structure["interfaces"]:
            parts.append(f"Interfaces: {', '.join(structure['interfaces'])}")
            parts.append("")

        if structure["types"]:
            parts.append(f"Types: {', '.join(structure['types'])}")
            parts.append("")

        if structure["enums"]:
            parts.append(f"Enums: {', '.join(structure['enums'])}")
            parts.append("")

        if structure["functions"]:
            parts.append(f"Functions ({len(structure['functions'])}):")
            for func in structure["functions"]:
                line_info = f" (line {func['line']})" if func.get("line") else ""
                parts.append(f"  {func['signature']}{line_info}")
                if func.get("docstring"):
                    doc_preview = func["docstring"][:200].replace("\n", " ")
                    parts.append(f"    Doc: {doc_preview}")
            parts.append("")

        docstrings = [c["text"] for c in structure["comments"] if c["is_docstring"]]
        if docstrings:
            parts.append(f"Documentation ({len(docstrings)} blocks):")
            for doc in docstrings[:5]:
                preview = doc[:300].replace("\n", " ").strip()
                parts.append(f"  {preview}")
            parts.append("")

        parts.append(f"Source code ({code.count(chr(10)) + 1} lines):")
        parts.append(code)

        return "\n".join(parts)