from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseChunker
from .recursive import RecursiveChunker
from .hierarchical import HierarchicalChunker
from .semantic import SemanticChunker
from .page import PageChunker
from .code import CodeChunker


class ChunkerFactory:

    _registry: Dict[str, Type[BaseChunker]] = {
        "recursive": RecursiveChunker,
        "hierarchical": HierarchicalChunker,
        "semantic": SemanticChunker,
        "page": PageChunker,
        "code": CodeChunker,
    }

    @classmethod
    def create(cls, strategy: str, config: Any = None) -> BaseChunker:
        key = strategy.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](config)

    @classmethod
    def register(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        cls._registry[name.lower()] = chunker_class

    @classmethod
    def list_strategies(cls) -> list:
        return sorted(cls._registry.keys())