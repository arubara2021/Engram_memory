from __future__ import annotations

import html as html_lib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


class MarkupParser(BaseParser):

    MARKUP_FORMATS = {"html", "htm", "xhtml", "md", "markdown", "mkd", "mdown", "rst", "rest", "xml", "svg"}

    def supports(self, format: str) -> bool:
        return format.lower() in self.MARKUP_FORMATS

    def parse(self, source: str) -> ParsedDocument:
        ext = os.path.splitext(source)[1].lower().lstrip(".")

        if not self._file_exists(source):
            if self._looks_like_markup(source):
                return self._parse_raw(source, ext or "text")
            raise FileNotFoundError(f"File not found: {source}")

        text = self._read_text(source)

        dispatch = {
            "html": self._parse_html,
            "htm": self._parse_html,
            "xhtml": self._parse_html,
            "xml": self._parse_xml,
            "svg": self._parse_xml,
            "md": self._parse_markdown,
            "markdown": self._parse_markdown,
            "mkd": self._parse_markdown,
            "mdown": self._parse_markdown,
            "rst": self._parse_rst,
            "rest": self._parse_rst,
        }

        handler = dispatch.get(ext, self._parse_html)
        return handler(text, source)

    def _looks_like_markup(self, text: str) -> bool:
        stripped = text.strip()[:200].lower()
        if stripped.startswith("<"):
            return True
        if re.search(r"^#{1,6}\s+", stripped, re.MULTILINE):
            return True
        if re.search(r"^[A-Z].*\n[=\-]{3,}", stripped, re.MULTILINE):
            return True
        return False

    def _parse_raw(self, text: str, format_hint: str) -> ParsedDocument:
        if format_hint in ("html", "htm", "xhtml", "xml", "svg"):
            return self._parse_html(text, "raw")
        if format_hint in ("md", "markdown", "mkd", "mdown"):
            return self._parse_markdown(text, "raw")
        if format_hint in ("rst", "rest"):
            return self._parse_rst(text, "raw")
        return self._parse_markdown(text, "raw")

    def _parse_html(self, text: str, source: str = "") -> ParsedDocument:
        text = self._strip_html_sections(text, ["script", "style", "noscript", "iframe", "object", "embed"])
        text = self._extract_html_structure(text)
        text = html_lib.unescape(text)
        text = self._normalize_whitespace(text)

        pages = [(1, text)]
        metadata = {
            "format": "html",
            "file_name": os.path.basename(source) if source else "",
        }

        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        return self._make_result(text, pages, metadata)

    def _strip_html_sections(self, html: str, tags: List[str]) -> str:
        for tag in tags:
            pattern = re.compile(
                rf"<{tag}[^>]*>.*?</{tag}>", re.DOTALL | re.IGNORECASE
            )
            html = pattern.sub("", html)
        return html

    def _extract_html_structure(self, html: str) -> str:
        heading_pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
        headings: Dict[str, str] = {}
        for match in heading_pattern.finditer(html):
            level = match.group(1).lower()
            content = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if content:
                num = level[1]
                prefix = "#" * int(num)
                headings[match.group(0)] = f"{prefix} {content}\n"

        for original, replacement in headings.items():
            html = html.replace(original, replacement)

        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</?p[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</?div[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</?li[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</?tr[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<td[^>]*>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<th[^>]*>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"</?blockquote[^>]*>", "\n", html, flags=re.IGNORECASE)

        html = re.sub(r"<[^>]+>", "", html)
        html = re.sub(r"&nbsp;", " ", html)
        html = re.sub(r"&[a-zA-Z]+;", " ", html)
        html = re.sub(r"&#\d+;", " ", html)
        html = re.sub(r"[ \t]+", " ", html)
        html = re.sub(r"\n{3,}", "\n\n", html)

        return html.strip()

    def _parse_markdown(self, text: str, source: str = "") -> ParsedDocument:
        text = self._strip_markdown_code_blocks(text)
        text = self._convert_markdown_headings(text)
        text = self._strip_markdown_formatting(text)
        text = self._convert_markdown_lists(text)
        text = self._convert_markdown_blockquotes(text)
        text = self._normalize_whitespace(text)

        pages = [(1, text)]
        metadata = {
            "format": "markdown",
            "file_name": os.path.basename(source) if source else "",
        }

        heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if heading_match:
            metadata["title"] = heading_match.group(1).strip()

        return self._make_result(text, pages, metadata)

    def _strip_markdown_code_blocks(self, text: str) -> str:
        def replace_code_block(match: Any) -> str:
            lang = match.group(1) or ""
            code = match.group(2)
            label = f" [{lang}]" if lang else ""
            return f"\n[Code{label}]\n{code.strip()}\n[/Code]\n"

        text = re.sub(
            r"```(\w*)\n(.*?)```", replace_code_block, text, flags=re.DOTALL
        )
        text = re.sub(r"~~~(\w*)\n(.*?)~~~", replace_code_block, text, flags=re.DOTALL)
        return text

    def _convert_markdown_headings(self, text: str) -> str:
        text = re.sub(r"^######\s+(.+)$", r"###### \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^#####\s+(.+)$", r"##### \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^####\s+(.+)$", r"#### \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^###\s+(.+)$", r"### \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^##\s+(.+)$", r"## \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^#\s+(.+)$", r"# \1\n", text, flags=re.MULTILINE)

        text = re.sub(r"^(.+)\n={3,}\s*$", r"# \1\n", text, flags=re.MULTILINE)
        text = re.sub(r"^(.+)\n-{3,}\s*$", r"## \1\n", text, flags=re.MULTILINE)

        return text

    def _strip_markdown_formatting(self, text: str) -> str:
        text = re.sub(r"$$([^$$]+)\]$$[^)]+$$", r"\1", text)
        text = re.sub(r"!$$([^$$]*)\]$$[^)]+$$", r"[Image: \1]", text)
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"___(.+?)___", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        text = re.sub(r"~~(.+?)~~", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.MULTILINE)
        text = re.sub(r"^(\s*)\d+\.\s+", r"\1- ", text, flags=re.MULTILINE)
        return text

    def _convert_markdown_lists(self, text: str) -> str:
        return text

    def _convert_markdown_blockquotes(self, text: str) -> str:
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        return text

    def _parse_rst(self, text: str, source: str = "") -> ParsedDocument:
        text = self._convert_rst_headings(text)
        text = self._strip_rst_directives(text)
        text = self._strip_rst_roles(text)
        text = self._strip_rst_references(text)
        text = self._normalize_whitespace(text)

        pages = [(1, text)]
        metadata = {
            "format": "rst",
            "file_name": os.path.basename(source) if source else "",
        }

        heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if heading_match:
            metadata["title"] = heading_match.group(1).strip()

        return self._make_result(text, pages, metadata)

    def _convert_rst_headings(self, text: str) -> str:
        lines = text.split("\n")
        result: List[str] = []
        i = 0

        overline_chars = set("=-~^\"'#*+")

        while i < len(lines):
            if i + 1 < len(lines) and i + 2 < len(lines):
                overline = lines[i].strip()
                title = lines[i + 1].strip()
                underline = lines[i + 2].strip()

                if (
                    overline
                    and title
                    and underline
                    and len(overline) == len(underline)
                    and len(overline) >= len(title)
                    and all(c == overline[0] for c in overline)
                    and overline[0] in overline_chars
                    and all(c == underline[0] for c in underline)
                    and underline[0] in overline_chars
                ):
                    level = self._rst_char_to_level(underline[0])
                    result.append(f"{'#' * level} {title}")
                    i += 3
                    continue

            if i + 1 < len(lines):
                title = lines[i].strip()
                underline = lines[i + 1].strip()

                if (
                    title
                    and underline
                    and len(underline) >= len(title)
                    and all(c == underline[0] for c in underline)
                    and underline[0] in overline_chars
                ):
                    level = self._rst_char_to_level(underline[0])
                    result.append(f"{'#' * level} {title}")
                    i += 2
                    continue

            result.append(lines[i])
            i += 1

        return "\n".join(result)

    def _rst_char_to_level(self, char: str) -> int:
        mapping = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "#": 1, "'": 5, "*": 3, "+": 4}
        return mapping.get(char, 3)

    def _strip_rst_directives(self, text: str) -> str:
        text = re.sub(
            r"^\.\.\s+\w+[^:]*::\s*.*?\n(\s+.*\n)*",
            "",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(r"^\.\.\s+$$.*?$$\s+.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\.\.\s+\w+:\s+.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\.\.\s+.*$", "", text, flags=re.MULTILINE)
        return text

    def _strip_rst_roles(self, text: str) -> str:
        text = re.sub(r":\w+:`([^`]+)`", r"\1", text)
        text = re.sub(r"`([^`]+)`_", r"\1", text)
        text = re.sub(r"\|([^|]+)\|", r"\1", text)
        return text

    def _strip_rst_references(self, text: str) -> str:
        text = re.sub(r"^\.\.\s+_\w+:\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:param\s+\w+:", " Parameter:", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:type\s+\w+:", " Type:", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:returns?:", " Returns:", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:rtype:", " Return type:", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:raises?\s+\w+:", " Raises:", text, flags=re.MULTILINE)
        text = re.sub(r"^\s+:example:", " Example:", text, flags=re.MULTILINE)
        return text

    def _parse_xml(self, text: str, source: str = "") -> ParsedDocument:
        cdata_pattern = re.compile(r"<!$$CDATA\[(.*?)$$\]>", re.DOTALL)
        cdata_blocks: List[str] = []
        for match in cdata_pattern.finditer(text):
            cdata_blocks.append(match.group(1))
        text = cdata_pattern.sub(" ", text)

        comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
        text = comment_pattern.sub("", text)

        proc_pattern = re.compile(r"<\?.*?\?>", re.DOTALL)
        text = proc_pattern.sub("", text)

        tag_content = re.findall(r">\s*([^<]+?)\s*<", text)
        text_parts = [t.strip() for t in tag_content if t.strip()]
        text_parts.extend(cdata_blocks)

        clean_text = "\n".join(text_parts)
        clean_text = html_lib.unescape(clean_text)
        clean_text = self._normalize_whitespace(clean_text)

        pages = [(1, clean_text)]
        metadata = {
            "format": "xml",
            "file_name": os.path.basename(source) if source else "",
        }

        return self._make_result(clean_text, pages, metadata)