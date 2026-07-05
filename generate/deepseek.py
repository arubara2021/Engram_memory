from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import LLMBackend


class DeepSeekBackend(LLMBackend):

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._api_key = self._get_config("llm_api_key", "DEEPSEEK_API_KEY")
        self._model = self._get_config("llm_model", "deepseek-chat")
        self._base_url = self._get_config("llm_base_url", "https://api.deepseek.com")

    def _get_config(self, attr: str, env_var: str) -> str:
        if self.config:
            val = getattr(self.config, attr, None)
            if val:
                return str(val)
        return os.environ.get(env_var, "")

    @property
    def model_name(self) -> str:
        return self._model or "deepseek-chat"

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
                "DeepSeek API key not set. "
                "Set llm_api_key in config or DEEPSEEK_API_KEY env var."
            )

        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/v1/chat/completions"

        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            choices = result.get("choices", [])
            if not choices:
                return ""

            return choices[0].get("message", {}).get("content", "")

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API connection error: {e.reason}")

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("DeepSeek API key not set.")

        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/v1/chat/completions"

        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API connection error: {e.reason}")

        choices = result.get("choices", [])
        text = choices[0].get("message", {}).get("content", "") if choices else ""

        usage = result.get("usage", {})

        return {
            "text": text,
            "model": result.get("model", self._model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": usage.get("prompt_tokens", self._count_tokens(prompt)),
            "response_tokens": usage.get("completion_tokens", self._count_tokens(text)),
            "total_tokens": usage.get("total_tokens", 0),
            "finish_reason": choices[0].get("finish_reason", "") if choices else "",
        }