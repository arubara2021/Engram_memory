from __future__ import annotations

import csv
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


class DataParser(BaseParser):

    DATA_FORMATS = {"json", "jsonl", "yaml", "yml", "csv", "tsv", "toml"}

    def supports(self, format: str) -> bool:
        return format.lower() in self.DATA_FORMATS

    def parse(self, source: str) -> ParsedDocument:
        if not self._file_exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        ext = os.path.splitext(source)[1].lower().lstrip(".")

        dispatch = {
            "json": self._parse_json,
            "jsonl": self._parse_json,
            "csv": self._parse_csv,
            "tsv": self._parse_csv,
            "yaml": self._parse_yaml,
            "yml": self._parse_yaml,
            "toml": self._parse_toml,
        }

        handler = dispatch.get(ext)
        if handler:
            return handler(source)

        raise ValueError(f"Unsupported data format: .{ext}")

    def _parse_json(self, path: str) -> ParsedDocument:
        text = self._read_text(path)
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".jsonl":
                return self._parse_jsonl(path)
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                return self._parse_jsonl(path)
            except Exception:
                return self._make_result(
                    text,
                    [(1, text)],
                    {"format": "json", "file_name": os.path.basename(path), "parse_error": True},
                )

        parts = self._json_to_text(data, "")
        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "json",
            "file_name": os.path.basename(path),
            "json_type": type(data).__name__,
        }
        if isinstance(data, list):
            metadata["item_count"] = len(data)
        elif isinstance(data, dict):
            metadata["key_count"] = len(data)

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _parse_jsonl(self, path: str) -> ParsedDocument:
        lines = self._read_text(path).strip().split("\n")
        parts: List[str] = []
        count = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj_parts = self._json_to_text(obj, f"Record {i + 1}")
                parts.append("\n".join(obj_parts))
                count += 1
            except json.JSONDecodeError:
                parts.append(f"Record {i + 1}: {line}")

        clean_text = "\n\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "jsonl",
            "file_name": os.path.basename(path),
            "record_count": count,
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _json_to_text(self, data: Any, prefix: str, depth: int = 0) -> List[str]:
        parts: List[str] = []
        indent = "  " * depth

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    parts.append(f"{indent}{key}:")
                    parts.extend(self._json_to_text(value, "", depth + 1))
                else:
                    parts.append(f"{indent}{key}: {self._format_value(value)}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    label = f"Item {i + 1}" if not prefix else f"{prefix} {i + 1}"
                    parts.append(f"{indent}{label}:")
                    parts.extend(self._json_to_text(item, "", depth + 1))
                else:
                    parts.append(f"{indent}- {self._format_value(item)}")
        else:
            parts.append(f"{indent}{self._format_value(data)}")

        return parts

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _parse_csv(self, path: str) -> ParsedDocument:
        text = self._read_text(path)
        ext = os.path.splitext(path)[1].lower()
        delimiter = "\t" if ext == ".tsv" else self._detect_csv_delimiter(text)

        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
        except Exception:
            return self._make_result(
                text,
                [(1, text)],
                {"format": "csv", "file_name": os.path.basename(path), "parse_error": True},
            )

        if not rows:
            return self._make_result(
                "",
                [],
                {"format": "csv", "file_name": os.path.basename(path)},
            )

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        parts: List[str] = []
        if headers:
            header_line = " | ".join(h.strip() for h in headers)
            parts.append(header_line)
            parts.append("-" * len(header_line))

        for row in data_rows:
            if all(not cell.strip() for cell in row):
                continue
            if headers and len(row) == len(headers):
                pairs = []
                for header, val in zip(headers, row):
                    val = val.strip()
                    if val:
                        pairs.append(f"{header.strip()}: {val}")
                if pairs:
                    parts.append(", ".join(pairs))
            else:
                cells = [cell.strip() for cell in row if cell.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "csv",
            "file_name": os.path.basename(path),
            "row_count": len(data_rows),
            "column_count": len(headers),
            "headers": [h.strip() for h in headers] if headers else [],
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _detect_csv_delimiter(self, text: str) -> str:
        sample = text[:4096]
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            first_line = sample.split("\n")[0] if sample else ""
            counts = {
                ",": first_line.count(","),
                ";": first_line.count(";"),
                "\t": first_line.count("\t"),
                "|": first_line.count("|"),
            }
            best = max(counts, key=lambda k: counts[k])
            return best if counts[best] > 0 else ","

    def _parse_yaml(self, path: str) -> ParsedDocument:
        text = self._read_text(path)

        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            return self._parse_yaml_fallback(text, path)
        except Exception:
            return self._make_result(
                text,
                [(1, text)],
                {"format": "yaml", "file_name": os.path.basename(path), "parse_error": True},
            )

        if data is None:
            data = {}

        parts = self._json_to_text(data, "")
        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "yaml",
            "file_name": os.path.basename(path),
            "yaml_type": type(data).__name__,
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _parse_yaml_fallback(self, text: str, path: str) -> ParsedDocument:
        parts: List[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                parts.append(stripped)
            else:
                parts.append(stripped)

        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "yaml",
            "file_name": os.path.basename(path),
            "fallback_parser": True,
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _parse_toml(self, path: str) -> ParsedDocument:
        text = self._read_text(path)

        try:
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
        except ImportError:
            try:
                import tomli

                with open(path, "rb") as f:
                    data = tomli.load(f)
            except ImportError:
                try:
                    import toml

                    data = toml.load(path)
                except ImportError:
                    return self._parse_toml_fallback(text, path)
        except Exception:
            return self._make_result(
                text,
                [(1, text)],
                {"format": "toml", "file_name": os.path.basename(path), "parse_error": True},
            )

        parts = self._json_to_text(data, "")
        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "toml",
            "file_name": os.path.basename(path),
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)

    def _parse_toml_fallback(self, text: str, path: str) -> ParsedDocument:
        parts: List[str] = []
        current_section = ""

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            section_match = re.match(r"^$$([^$$]+)\]$", stripped)
            if section_match:
                current_section = section_match.group(1)
                parts.append(f"\n{current_section}:")
                continue

            if "=" in stripped:
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip()
                value = value.strip('"').strip("'")
                if key:
                    parts.append(f"  {key}: {value}")
            else:
                parts.append(f"  {stripped}")

        clean_text = "\n".join(parts)
        clean_text = self._normalize_whitespace(clean_text)

        metadata = {
            "format": "toml",
            "file_name": os.path.basename(path),
            "fallback_parser": True,
        }

        pages = [(1, clean_text)]
        return self._make_result(clean_text, pages, metadata)