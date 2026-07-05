from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBackend(ABC):

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    def dimension(self) -> int:
        return 384