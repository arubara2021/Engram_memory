from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core.types import Chunk

from .base import EmbeddingBackend
from .local import LocalEmbedder
from .remote import RemoteEmbedder
from .store import VectorStore


class EmbeddingPipeline:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.backend: Optional[EmbeddingBackend] = None
        self.store: VectorStore = VectorStore(
            dimension=getattr(config, "embedding_dimension", 384) if config else 384
        )
        self._initialized = False
        self._auto_init()

    def _auto_init(self) -> None:
        if not self.config:
            return
        if not getattr(self.config, "embedding_enabled", False):
            return

        backend_type = getattr(self.config, "embedding_backend", "local")
        model = getattr(self.config, "embedding_model", "all-MiniLM-L6-v2")
        dimension = getattr(self.config, "embedding_dimension", 384)

        if backend_type == "local":
            embedder = LocalEmbedder(model_name=model, dimension=dimension)
            if embedder.load():
                self.backend = embedder
                self.store = VectorStore(dimension=embedder.dimension)
                self._initialized = True
        elif backend_type == "remote":
            api_url = getattr(self.config, "llm_base_url", None) or ""
            api_key = getattr(self.config, "llm_api_key", None) or ""
            embedder = RemoteEmbedder(
                api_url=api_url,
                api_key=api_key,
                model=model,
                dimension=dimension,
            )
            if embedder.is_available():
                self.backend = embedder
                self.store = VectorStore(dimension=dimension)
                self._initialized = True

    def is_available(self) -> bool:
        return self.backend is not None and self.backend.is_available()

    def set_backend(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.store = VectorStore(dimension=backend.dimension)
        self._initialized = True

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not self.is_available():
            return chunks

        texts = [c.text for c in chunks]
        batch_size = getattr(self.config, "embedding_batch_size", 32) if self.config else 32

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.backend.embed(batch)
            all_vectors.extend(vectors)

        for chunk, vector in zip(chunks, all_vectors):
            chunk.embedding = vector
            self.store.add(chunk.chunk_id, vector, {"doc_id": chunk.doc_id})

        return chunks

    def embed_query(self, query_text: str) -> List[float]:
        if not self.is_available():
            return []
        return self.backend.embed_single(query_text)

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embed_query(query_text)
        if not query_vector:
            return []
        return self.store.search(query_vector, top_k=top_k, threshold=threshold)

    def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        return self.store.search(query_vector, top_k=top_k, threshold=threshold)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "backend_type": type(self.backend).__name__ if self.backend else None,
            "dimension": self.store.dimension,
            "stored_vectors": self.store.size(),
            "initialized": self._initialized,
        }