from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .base import LLMBackend
from .deepseek import DeepSeekBackend
from .openai import OpenAIBackend
from .ollama import OllamaBackend
from .anthropic import AnthropicBackend


class LLMFactory:

    _registry: Dict[str, Type[LLMBackend]] = {
        "deepseek": DeepSeekBackend,
        "openai": OpenAIBackend,
        "ollama": OllamaBackend,
        "anthropic": AnthropicBackend,
    }

    @classmethod
    def create(cls, backend_name: str, config: Any = None) -> LLMBackend:
        key = backend_name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown LLM backend: '{backend_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](config)

    @classmethod
    def register(cls, name: str, backend_class: Type[LLMBackend]) -> None:
        cls._registry[name.lower()] = backend_class

    @classmethod
    def list_backends(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def get_available(cls, config: Any = None) -> List[str]:
        available: List[str] = []
        for name, backend_cls in cls._registry.items():
            try:
                backend = backend_cls(config)
                if backend.is_available():
                    available.append(name)
            except Exception:
                pass
        return available

    @classmethod
    def create_best_available(cls, config: Any = None) -> Optional[LLMBackend]:
        priority = ["deepseek", "openai", "anthropic", "ollama"]
        for name in priority:
            try:
                backend = cls.create(name, config)
                if backend.is_available():
                    return backend
            except Exception:
                continue
        return None