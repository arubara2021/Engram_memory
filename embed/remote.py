from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class RemoteEmbedder:

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 384,
    ) -> None:
        self.api_url = api_url or os.environ.get("EMBEDDING_API_URL", "")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        self.model = model
        self._dimension = dimension
        self._is_available: Optional[bool] = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        if self._is_available is not None:
            return self._is_available
        if not self.api_url or not self.api_key:
            self._is_available = False
            return False
        self._is_available = True
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available():
            raise RuntimeError(
                "Remote embedder not configured. Set api_url and api_key."
            )

        if self._is_openai_compatible():
            return self._embed_openai(texts)

        return self._embed_generic(texts)

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _is_openai_compatible(self) -> bool:
        url = self.api_url.lower()
        return "openai" in url or "api.openai.com" in url

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = json.dumps({
                "input": batch,
                "model": self.model,
                "encoding_format": "float",
            }).encode("utf-8")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            url = self.api_url.rstrip("/")
            if not url.endswith("/embeddings"):
                url = url + "/v1/embeddings"

            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                data = result.get("data", [])
                sorted_data = sorted(data, key=lambda x: x.get("index", 0))
                for item in sorted_data:
                    all_embeddings.append(item.get("embedding", []))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Embedding API error {e.code}: {body}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"Embedding API connection error: {e.reason}")

        return all_embeddings

    def _embed_generic(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = json.dumps({
                "texts": batch,
                "model": self.model,
            }).encode("utf-8")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            try:
                req = urllib.request.Request(
                    self.api_url, data=payload, headers=headers
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                embeddings = result.get("embeddings", result.get("data", []))
                if isinstance(embeddings, list):
                    for item in embeddings:
                        if isinstance(item, list):
                            all_embeddings.append(item)
                        elif isinstance(item, dict):
                            all_embeddings.append(item.get("embedding", []))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Embedding API error {e.code}: {body}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"Embedding API connection error: {e.reason}")

        return all_embeddings