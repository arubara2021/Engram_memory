from __future__ import annotations

import email
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


class TextParser(BaseParser):

    TEXT_FORMATS = {"txt", "text", "log", "eml", "dat", "cfg", "ini", "conf", "properties", "env", "gitignore", "md5", "sha256"}

    def supports(self, format: str) -> bool:
        return format.lower() in self.TEXT_FORMATS

    def parse(self, source: str) -> ParsedDocument:
        if not self._file_exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        ext = os.path.splitext(source)[1].lower().lstrip(".")
        file_name = os.path.basename(source).lower()

        if ext == "eml" or file_name.endswith(".eml"):
            return self._parse_eml(source)
        if ext == "log" or "log" in file_name:
            return self._parse_log(source)
        return self._parse_plain(source)

    def _parse_plain(self, path: str) -> ParsedDocument:
        text = self._read_text(path)
        text = self._strip_control_chars(text)
        text = self._normalize_whitespace(text)

        pages = [(1, text)]
        metadata = {
            "format": "text",
            "file_name": os.path.basename(path),
            "line_count": text.count("\n") + 1,
            "encoding_detected": True,
        }

        encoding = self._detect_encoding(path)
        if encoding:
            metadata["encoding"] = encoding

        return self._make_result(text, pages, metadata)

    def _parse_log(self, path: str) -> ParsedDocument:
        text = self._read_text(path)
        text = self._strip_control_chars(text)

        entries = self._parse_log_entries(text)
        readable_parts: List[str] = []

        if entries:
            for entry in entries[:5000]:
                parts = []
                if entry.get("timestamp"):
                    parts.append(f"[{entry['timestamp']}]")
                if entry.get("level"):
                    parts.append(f"[{entry['level']}]")
                if entry.get("message"):
                    parts.append(entry["message"])
                if parts:
                    readable_parts.append(" ".join(parts))

            if len(entries) > 5000:
                readable_parts.append(
                    f"\n... {len(entries) - 5000} more log entries omitted ..."
                )

        if not readable_parts:
            readable_parts = [text]

        clean_text = "\n".join(readable_parts)
        clean_text = self._normalize_whitespace(clean_text)

        pages = [(1, clean_text)]

        levels = {}
        for entry in entries:
            level = entry.get("level", "UNKNOWN")
            levels[level] = levels.get(level, 0) + 1

        metadata = {
            "format": "log",
            "file_name": os.path.basename(path),
            "entry_count": len(entries),
            "log_levels": levels,
            "line_count": text.count("\n") + 1,
        }

        return self._make_result(clean_text, pages, metadata)

    def _parse_log_entries(self, text: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []

        timestamp_pattern = re.compile(
            r"(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s*[+-]\d{2}:?\d{2})?)"
        )
        level_pattern = re.compile(
            r"\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|TRACE|NOTICE|ALERT|EMERGENCY)\b",
            re.IGNORECASE,
        )

        current_entry: Dict[str, str] = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                continue

            ts_match = timestamp_pattern.search(line)
            level_match = level_pattern.search(line)

            if ts_match or level_match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = {}
                if ts_match:
                    current_entry["timestamp"] = ts_match.group(1).strip()
                if level_match:
                    current_entry["level"] = level_match.group(1).upper()
                    if current_entry["level"].startswith("WARN"):
                        current_entry["level"] = "WARNING"
                remaining = line
                if ts_match:
                    remaining = remaining.replace(ts_match.group(1), "")
                if level_match:
                    remaining = remaining.replace(level_match.group(0), "")
                remaining = remaining.strip(" \t[]():")
                current_entry["message"] = remaining
            elif current_entry:
                if "message" in current_entry:
                    current_entry["message"] += "\n" + line
                else:
                    current_entry["message"] = line

        if current_entry:
            entries.append(current_entry)

        return entries

    def _parse_eml(self, path: str) -> ParsedDocument:
        raw = self._read_bytes(path)

        try:
            msg = email.message_from_bytes(raw)
        except Exception:
            text = raw.decode("utf-8", errors="replace")
            return self._make_result(
                text,
                [(1, text)],
                {"format": "eml", "file_name": os.path.basename(path), "parse_error": True},
            )

        parts: List[str] = []

        headers_to_extract = [
            "From", "To", "Cc", "Bcc", "Subject", "Date",
            "Reply-To", "Message-ID",
        ]
        for header in headers_to_extract:
            value = msg.get(header)
            if value:
                parts.append(f"{header}: {value}")

        parts.append("")

        body_text = self._extract_email_body(msg)
        if body_text:
            parts.append(body_text)

        attachments = self._extract_email_attachments(msg)
        if attachments:
            parts.append(f"\nAttachments ({len(attachments)}):")
            for att in attachments:
                parts.append(f"  - {att['filename']} ({att['content_type']}, {att['size']} bytes)")

        clean_text = "\n".join(parts)
        clean_text = self._strip_control_chars(clean_text)
        clean_text = self._normalize_whitespace(clean_text)

        pages = [(1, clean_text)]

        metadata = {
            "format": "eml",
            "file_name": os.path.basename(path),
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
            "has_attachments": len(attachments) > 0,
            "attachment_count": len(attachments),
        }

        return self._make_result(clean_text, pages, metadata)

    def _extract_email_body(self, msg: Any) -> str:
        body_parts: List[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in disposition:
                    continue

                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            text = payload.decode(charset, errors="replace")
                            body_parts.append(text)
                        except Exception:
                            body_parts.append(payload.decode("utf-8", errors="replace"))

                elif content_type == "text/html" and not body_parts:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            html_text = payload.decode(charset, errors="replace")
                            plain = self._strip_html_simple(html_text)
                            body_parts.append(plain)
                        except Exception:
                            pass
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                    if content_type == "text/html":
                        text = self._strip_html_simple(text)
                    body_parts.append(text)
                except Exception:
                    body_parts.append(payload.decode("utf-8", errors="replace"))

        return "\n".join(body_parts)

    def _extract_email_attachments(self, msg: Any) -> List[Dict[str, Any]]:
        attachments: List[Dict[str, Any]] = []
        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                filename = part.get_filename() or "unnamed"
                content_type = part.get_content_type() or "application/octet-stream"
                payload = part.get_payload(decode=True)
                size = len(payload) if payload else 0
                attachments.append({
                    "filename": filename,
                    "content_type": content_type,
                    "size": size,
                })

        return attachments

    def _strip_html_simple(self, html: str) -> str:
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</?p[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "", html)
        html = html.replace("&nbsp;", " ").replace("&amp;", "&")
        html = html.replace("&lt;", "<").replace("&gt;", ">")
        html = html.replace("&quot;", '"')
        html = re.sub(r"&#?\w+;", " ", html)
        html = re.sub(r"\n{3,}", "\n\n", html)
        return html.strip()

    def _detect_encoding(self, path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                raw = f.read(4)

            if raw[:3] == b"\xef\xbb\xbf":
                return "utf-8-bom"
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                return "utf-16"
            if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
                return "utf-32"

            try:
                import chardet

                with open(path, "rb") as f:
                    raw = f.read(65536)
                detected = chardet.detect(raw)
                if detected and detected.get("encoding"):
                    return detected["encoding"].lower()
            except ImportError:
                pass

            return None
        except Exception:
            return None