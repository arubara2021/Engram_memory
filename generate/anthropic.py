from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import LLMBackend


class AnthropicBackend(LLMBackend):

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._api_key = self._get_config("llm_api_key", "ANTHROPIC_API_KEY")
        self._model = self._get_config("llm_model", "claude-sonnet-4-20250514")
        self._base_url = self._get_config("llm_base_url", "https://api.anthropic.com")
        self._version = "2023-06-01"

    def _get_config(self, attr: str, env_var: str) -> str:
        if self.config:
            val = getattr(self.config, attr, None)
            if val:
                return str(val)
        return os.environ.get(env_var, "")

    @property
    def model_name(self) -> str:
        return self._model or "claude-sonnet-4-20250514"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Anthropic API key not set. "
                "Set llm_api_key in config or ANTHROPIC_API_KEY env var."
            )

        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/v1/messages"

        payload = json.dumps({
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            content = result.get("content", [])
            if not content:
                return ""

            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))

            return "\n".join(parts)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anthropic API connection error: {e.reason}")

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Anthropic API key not set.")

        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/v1/messages"

        payload = json.dumps({
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anthropic API connection error: {e.reason}")

        content = result.get("content", [])
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(parts)

        usage = result.get("usage", {})

        return {
            "text": text,
            "model": result.get("model", self._model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": usage.get("input_tokens", self._count_tokens(prompt)),
            "response_tokens": usage.get("output_tokens", self._count_tokens(text)),
            "stop_reason": result.get("stop_reason", ""),
        }