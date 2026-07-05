from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser
from ..core.types import ParsedDocument


IMAGE_FORMATS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp", "ico"}


class ImageParser(BaseParser):

    def supports(self, format: str) -> bool:
        return format.lower() in IMAGE_FORMATS

    def parse(self, source: str) -> ParsedDocument:
        if not self._file_exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")

        image_info = self._get_image_info(source)
        text = self._ocr_image(source)
        text = self._strip_control_chars(text)
        text = self._normalize_whitespace(text)

        pages = [(1, text)] if text else []
        metadata = {
            "format": "image",
            "file_name": os.path.basename(source),
            "image_format": image_info.get("format", ""),
            "image_width": image_info.get("width", 0),
            "image_height": image_info.get("height", 0),
            "ocr_performed": True,
            "ocr_text_length": len(text),
        }

        result = self._make_result(text, pages, metadata)
        if not text:
            result.text_quality = 0.0
        return result

    def _get_image_info(self, path: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        info["format"] = ext

        try:
            from PIL import Image

            with Image.open(path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["mode"] = img.mode
                info["format"] = img.format or ext
        except ImportError:
            pass
        except Exception:
            pass

        return info

    def _ocr_image(self, path: str) -> str:
        text = self._ocr_tesseract(path)
        if text:
            return text

        text = self._ocr_easyocr(path)
        if text:
            return text

        text = self._ocr_pytesseract(path)
        if text:
            return text

        return ""

    def _ocr_tesseract(self, path: str) -> str:
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            img.close()
            return text.strip()
        except ImportError:
            return ""
        except Exception:
            return ""

    def _ocr_easyocr(self, path: str) -> str:
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            results = reader.readtext(path, detail=0)
            return "\n".join(results).strip()
        except ImportError:
            return ""
        except Exception:
            return ""

    def _ocr_pytesseract(self, path: str) -> str:
        try:
            import subprocess
            import shutil

            if shutil.which("tesseract") is None:
                return ""

            result = subprocess.run(
                ["tesseract", path, "stdout"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
        except Exception:
            return ""