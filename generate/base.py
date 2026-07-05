from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBackend(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        text = self.generate(prompt, temperature, max_tokens)
        return {
            "text": text,
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": self._count_tokens(prompt),
            "response_tokens": self._count_tokens(text),
        }

    @property
    def model_name(self) -> str:
        return "unknown"

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def _truncate_prompt(self, prompt: str, max_chars: int = 120000) -> str:
        if len(prompt) <= max_chars:
            return prompt
        return prompt[:max_chars] + "\n\n[Prompt truncated]"