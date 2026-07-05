from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker
from ..core.types import Chunk


EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    "py": "python", "pyw": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript", "jsx": "javascript",
    "java": "java",
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "rs": "rust",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin", "kts": "kotlin",
    "scala": "scala",
    "r": "r", "R": "r",
    "sh": "shell", "bash": "shell", "zsh": "shell",
    "sql": "sql",
    "css": "css", "scss": "scss",
    "lua": "lua",
    "dart": "dart",
    "zig": "zig",
    "nim": "nim",
    "jl": "julia",
    "vue": "vue",
    "svelte": "svelte",
    "html": "html",
    "xml": "xml",
}

PYTHON_DEF = re.compile(
    r"^((?:async\s+)?def\s+\w+\s*$$[^)]*$$(?:\s*->\s*[^:]+)?)\s*:",
    re.MULTILINE,
)
PYTHON_CLASS = re.compile(
    r"^(class\s+\w+(?:$$[^)]*$$)?\s*:)",
    re.MULTILINE,
)
GENERAL_FUNC = re.compile(
    r"^((?:export\s+)?(?:async\s+)?function\s+\w+\s*$$[^)]*$$[^{]*)",
    re.MULTILINE,
)
GENERAL_CLASS = re.compile(
    r"^((?:export\s+)?(?:abstract\s+)?class\s+\w+[^{]*)",
    re.MULTILINE,
)
GENERAL_INTERFACE = re.compile(
    r"^((?:export\s+)?interface\s+\w+[^{]*)",
    re.MULTILINE,
)
GENERAL_ENUM = re.compile(
    r"^((?:export\s+)?enum\s+\w+[^{]*)",
    re.MULTILINE,
)
GENERAL_TYPE = re.compile(
    r"^((?:export\s+)?type\s+\w+[^=]*)",
    re.MULTILINE,
)
GENERAL_STRUCT = re.compile(
    r"^(pub\s+)?(struct\s+\w+[^{]*)",
    re.MULTILINE,
)
METHOD_PATTERN = re.compile(
    r"^\s{2,}((?:pub|priv|prot|internal|static|async|virtual|override|abstract|open|final|get|set)\s+)*"
    r"(?:func|def|fn|fun|method|procedure|proc)\s+(\w+)\s*$$",
    re.MULTILINE,
)
GO_FUNC = re.compile(
    r"^func\s+(?:\([^)]+$$\s+)?(\w+)\s*$$[^)]*$$[^{]*",
    re.MULTILINE,
)
RUST_FUNC = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)[^{]*",
    re.MULTILINE,
)
C_FUNC = re.compile(
    r"^(?:(?:static|inline|extern|const|void|int|float|double|char|long|short|unsigned|signed|bool|size_t|ssize_t|uint\d+_t|int\d+_t|struct\s+\w+|enum\s+\w+|typedef)\s+)*"
    r"(\w+)\s*$$[^)]*$$\s*\{",
    re.MULTILINE,
)
RUBY_DEF = re.compile(
    r"^\s*(?:def|class|module)\s+(\w[\w!?]*)",
    re.MULTILINE,
)


class CodeChunker(BaseChunker):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._max_chunk_lines = 150
        self._min_chunk_lines = 3

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        language = "unknown"
        if metadata:
            language = metadata.get("language", "unknown")
            if language == "unknown":
                ext = metadata.get("file_name", "")
                if "." in ext:
                    ext = ext.rsplit(".", 1)[-1].lower()
                    language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

        blocks = self._extract_blocks(text, language)
        chunks: List[Chunk] = []
        idx = 0

        if blocks:
            header = self._build_header(text, blocks, metadata)
            imports = self._extract_imports(text, language)
            if imports or header:
                preamble_parts = []
                if header:
                    preamble_parts.append(header)
                if imports:
                    preamble_parts.append(imports)
                preamble = "\n\n".join(preamble_parts)
                if self._word_count(preamble) >= 5:
                    chunks.append(self._make_chunk(
                        preamble, f"code_{idx}", doc_id, "header", idx
                    ))
                    idx += 1

            for block_type, name, block_text, line_num in blocks:
                block_wc = self._word_count(block_text)

                if block_wc > self._max_words * 2:
                    sub_chunks = self._split_large_block(
                        block_text, name, block_type, doc_id, idx
                    )
                    chunks.extend(sub_chunks)
                    idx += len(sub_chunks)
                elif block_wc >= 5:
                    section_label = f"{block_type}:{name}" if name else block_type
                    chunks.append(self._make_chunk(
                        block_text,
                        f"code_{idx}",
                        doc_id,
                        section_label,
                        idx,
                    ))
                    idx += 1

            remaining = self._extract_remaining(text, blocks)
            if remaining.strip() and self._word_count(remaining) >= self._min_words:
                chunks.append(self._make_chunk(
                    remaining.strip(), f"code_{idx}", doc_id, "remaining", idx
                ))
                idx += 1
        else:
            chunks = self._fallback_line_split(text, doc_id)

        return chunks

    def _extract_blocks(
        self, code: str, language: str
    ) -> List[Tuple[str, str, str, int]]:
        patterns = self._get_patterns(language)
        if not patterns:
            return self._generic_block_split(code)

        candidates: List[Tuple[int, str, str, re.Match]] = []
        for pattern, block_type in patterns:
            for match in pattern.finditer(code):
                name = ""
                for g in match.groups():
                    if g and re.match(r"^[a-zA-Z_]\w*$", g.strip()):
                        name = g.strip()
                        break
                if not name:
                    sig = match.group(0).strip()
                    name_match = re.search(r"(\w+)\s*\(", sig)
                    if name_match:
                        name = name_match.group(1)
                candidates.append((match.start(), block_type, name, match))

        candidates.sort(key=lambda x: x[0])

        blocks: List[Tuple[str, str, str, int]] = []
        for i, (start, block_type, name, match) in enumerate(candidates):
            end = candidates[i + 1][0] if i + 1 < len(candidates) else len(code)
            block_text = code[start:end].rstrip()
            if not block_text.strip():
                continue
            line_num = code[:start].count("\n") + 1
            blocks.append((block_type, name, block_text, line_num))

        return blocks

    def _get_patterns(self, language: str) -> List[Tuple[re.Pattern, str]]:
        if language == "python":
            return [
                (PYTHON_CLASS, "class"),
                (PYTHON_DEF, "function"),
            ]
        if language in ("javascript", "typescript", "vue", "svelte"):
            return [
                (GENERAL_CLASS, "class"),
                (GENERAL_INTERFACE, "interface"),
                (GENERAL_ENUM, "enum"),
                (GENERAL_TYPE, "type"),
                (GENERAL_FUNC, "function"),
            ]
        if language == "go":
            return [(GO_FUNC, "function")]
        if language == "rust":
            return [
                (GENERAL_STRUCT, "struct"),
                (GENERAL_ENUM, "enum"),
                (RUST_FUNC, "function"),
            ]
        if language in ("ruby",):
            return [(RUBY_DEF, "definition")]
        if language in ("c", "cpp"):
            return [
                (GENERAL_STRUCT, "struct"),
                (GENERAL_ENUM, "enum"),
                (C_FUNC, "function"),
            ]
        if language in ("java", "csharp", "kotlin", "scala", "swift"):
            return [
                (GENERAL_CLASS, "class"),
                (GENERAL_INTERFACE, "interface"),
                (GENERAL_ENUM, "enum"),
            ]
        if language in ("css", "scss"):
            return self._css_patterns()
        if language == "html":
            return []
        if language == "sql":
            return self._sql_patterns()
        return [
            (GENERAL_CLASS, "class"),
            (GENERAL_FUNC, "function"),
        ]

    def _css_patterns(self) -> List[Tuple[re.Pattern, str]]:
        return [
            (re.compile(r"^(@(?:media|keyframes|font-face|supports|layer)\s+[^{]*\{)", re.MULTILINE), "at_rule"),
            (re.compile(r"^([.#]?\w[\w\s.,:>+~#\[\]=]*\s*\{)", re.MULTILINE), "rule"),
        ]

    def _sql_patterns(self) -> List[Tuple[re.Pattern, str]]:
        return [
            (re.compile(r"^(CREATE\s+(?:TABLE|VIEW|INDEX|FUNCTION|PROCEDURE|TRIGGER|SCHEMA|DATABASE)\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w.]+)", re.MULTILINE | re.IGNORECASE), "definition"),
            (re.compile(r"^(ALTER\s+TABLE\s+[\w.]+)", re.MULTILINE | re.IGNORECASE), "alteration"),
            (re.compile(r"^(INSERT\s+INTO\s+[\w.]+)", re.MULTILINE | re.IGNORECASE), "insert"),
            (re.compile(r"^(SELECT\b.*?\bFROM\b)", re.MULTILINE | re.IGNORECASE), "select"),
        ]

    def _generic_block_split(self, code: str) -> List[Tuple[str, str, str, int]]:
        lines = code.split("\n")
        blocks: List[Tuple[str, str, str, int]] = []
        current_lines: List[str] = []
        current_type = ""
        current_name = ""
        current_start = 0
        brace_depth = 0
        in_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not in_block:
                is_block_start = False
                for pattern, btype in [
                    (re.compile(r"^(?:class|struct|enum|interface)\b"), "block"),
                    (re.compile(r"^(?:function|def|fn|func|pub fn|pub func)\b"), "function"),
                    (re.compile(r"^(?:public|private|protected|static|async)\s+"), "method"),
                ]:
                    if pattern.match(stripped):
                        is_block_start = True
                        current_type = btype
                        name_match = re.search(r"(\w+)\s*[\({:]", stripped)
                        current_name = name_match.group(1) if name_match else ""
                        break

                if is_block_start:
                    if current_lines and "\n".join(current_lines).strip():
                        text = "\n".join(current_lines)
                        if self._word_count(text) >= 5:
                            blocks.append((current_type or "block", current_name, text, current_start + 1))
                    current_lines = [line]
                    current_start = i
                    in_block = True
                    brace_depth = line.count("{") - line.count("}")
                    if brace_depth <= 0 and "{" not in line:
                        brace_depth = 0
                    continue

            if in_block:
                current_lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                if (brace_depth <= 0 and "{" in "\n".join(current_lines)) or (
                    not any("{" in cl for cl in current_lines) and
                    (not stripped or (stripped and not stripped.startswith((" ", "\t")) and current_lines))
                ):
                    text = "\n".join(current_lines)
                    if self._word_count(text) >= 5:
                        blocks.append((current_type or "block", current_name, text, current_start + 1))
                    current_lines = []
                    in_block = False
                    brace_depth = 0
            else:
                current_lines.append(line)

        if current_lines:
            text = "\n".join(current_lines)
            if text.strip() and self._word_count(text) >= 5:
                blocks.append((current_type or "block", current_name, text, current_start + 1))

        return blocks

    def _build_header(
        self, code: str, blocks: List[Tuple[str, str, str, int]], metadata: Optional[Dict]
    ) -> str:
        parts: List[str] = []
        file_name = ""
        if metadata:
            file_name = metadata.get("file_name", "")
        if file_name:
            parts.append(f"File: {file_name}")

        if blocks:
            structure_parts: List[str] = []
            classes = [b for b in blocks if b[0] == "class"]
            functions = [b for b in blocks if b[0] in ("function", "method", "definition")]
            others = [b for b in blocks if b[0] not in ("class", "function", "method", "definition")]

            if classes:
                structure_parts.append(f"Classes: {', '.join(b[1] for b in classes if b[1])}")
            if functions:
                names = [b[1] for b in functions if b[1]]
                if len(names) > 10:
                    structure_parts.append(f"Functions ({len(names)}): {', '.join(names[:10])}, ...")
                else:
                    structure_parts.append(f"Functions: {', '.join(names)}")
            if others:
                structure_parts.append(f"Other blocks: {len(others)}")

            if structure_parts:
                parts.append("Structure: " + " | ".join(structure_parts))

        return "\n".join(parts)

    def _extract_imports(self, code: str, language: str) -> str:
        import_patterns: Dict[str, re.Pattern] = {
            "python": re.compile(r"^(?:from\s+\S+\s+)?import\s+.+$", re.MULTILINE),
            "javascript": re.compile(r"^import\s+.+from\s+['\"]|^(?:const|let|var)\s+\w+\s*=\s*require\(", re.MULTILINE),
            "typescript": re.compile(r"^import\s+.+from\s+['\"]", re.MULTILINE),
            "java": re.compile(r"^import\s+[\w.]+;?$", re.MULTILINE),
            "go": re.compile(r'^import\s+(?:"[^"]+"|\()', re.MULTILINE),
            "rust": re.compile(r"^use\s+[\w:]+;?$", re.MULTILINE),
            "c": re.compile(r"^#\s*include\s+[<\"].+[>\"]$", re.MULTILINE),
            "cpp": re.compile(r"^#\s*include\s+[<\"].+[>\"]$", re.MULTILINE),
            "ruby": re.compile(r"^(?:require|require_relative)\s+['\"]", re.MULTILINE),
            "php": re.compile(r"^use\s+[\w\\]+;?$|^require(?:_once)?\s", re.MULTILINE),
            "swift": re.compile(r"^import\s+\w+$", re.MULTILINE),
            "kotlin": re.compile(r"^import\s+[\w.]+$", re.MULTILINE),
            "scala": re.compile(r"^import\s+[\w._]+$", re.MULTILINE),
        }

        pattern = import_patterns.get(language)
        if not pattern:
            return ""

        matches = pattern.findall(code)
        if not matches:
            return ""

        lines = [m.strip() for m in matches if m.strip()]
        if not lines:
            return ""

        section = f"Imports ({len(lines)}):\n" + "\n".join(f"  {l}" for l in lines[:30])
        if len(lines) > 30:
            section += f"\n  ... and {len(lines) - 30} more"

        return section

    def _split_large_block(
        self, text: str, name: str, block_type: str, doc_id: str, start_idx: int
    ) -> List[Chunk]:
        lines = text.split("\n")
        chunks: List[Chunk] = []
        current_lines: List[str] = []
        current_wc = 0
        idx = start_idx

        for line in lines:
            line_wc = len(line.split())
            if current_wc + line_wc > self._max_words and current_lines:
                combined = "\n".join(current_lines)
                if current_wc >= self._min_words:
                    section_label = f"{block_type}:{name}" if name else block_type
                    chunks.append(self._make_chunk(
                        combined, f"code_{idx}", doc_id, section_label, idx
                    ))
                    idx += 1
                current_lines = []
                current_wc = 0
            current_lines.append(line)
            current_wc += line_wc

        if current_lines:
            combined = "\n".join(current_lines)
            if current_wc >= self._min_words:
                section_label = f"{block_type}:{name}" if name else block_type
                chunks.append(self._make_chunk(
                    combined, f"code_{idx}", doc_id, section_label, idx
                ))

        return chunks

    def _extract_remaining(
        self, code: str, blocks: List[Tuple[str, str, str, int]]
    ) -> str:
        if not blocks:
            return code

        block_texts = {b[2] for b in blocks}
        lines = code.split("\n")
        remaining_lines: List[str] = []
        in_block = False

        for line in lines:
            if any(line in bt for bt in block_texts):
                in_block = True
                continue
            if in_block:
                for bt in block_texts:
                    if line in bt:
                        continue
                in_block = False
            if not in_block:
                remaining_lines.append(line)

        return "\n".join(remaining_lines)

    def _fallback_line_split(self, code: str, doc_id: str) -> List[Chunk]:
        lines = code.split("\n")
        chunks: List[Chunk] = []
        current_lines: List[str] = []
        current_wc = 0
        idx = 0

        for line in lines:
            line_wc = len(line.split())
            if current_wc + line_wc > self._max_words and current_lines:
                combined = "\n".join(current_lines)
                if current_wc >= self._min_words:
                    chunks.append(self._make_chunk(
                        combined, f"code_{idx}", doc_id, "", idx
                    ))
                    idx += 1
                current_lines = []
                current_wc = 0
            current_lines.append(line)
            current_wc += line_wc

        if current_lines:
            combined = "\n".join(current_lines)
            if combined.strip() and current_wc >= self._min_words:
                chunks.append(self._make_chunk(
                    combined, f"code_{idx}", doc_id, "", idx
                ))

        return chunks