from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import LLMBackend


class OllamaBackend(LLMBackend):

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._base_url = self._get_config("llm_base_url", "OLLAMA_HOST")
        if not self._base_url:
            self._base_url = "http://localhost:11434"
        self._model = self._get_config("llm_model", "OLLAMA_MODEL")
        if not self._model:
            self._model = "llama3"

    def _get_config(self, attr: str, env_var: str) -> str:
        if self.config:
            val = getattr(self.config, attr, None)
            if val:
                return str(val)
        return os.environ.get(env_var, "")

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            url = self._base_url.rstrip("/") + "/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in result.get("models", [])]
                target = self._model
                for m in models:
                    if m == target or m.startswith(target + ":"):
                        return True
                if models:
                    return True
                return False
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/api/generate"

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            return result.get("response", "")

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama connection error: {e.reason}. "
                "Make sure Ollama is running (ollama serve)."
            )

    def generate_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/api/chat"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            message = result.get("message", {})
            return message.get("content", "")

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama connection error: {e.reason}. "
                "Make sure Ollama is running (ollama serve)."
            )

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        prompt = self._truncate_prompt(prompt)

        url = self._base_url.rstrip("/") + "/api/generate"

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama connection error: {e.reason}. "
                "Make sure Ollama is running (ollama serve)."
            )

        text = result.get("response", "")

        return {
            "text": text,
            "model": result.get("model", self._model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": result.get("prompt_eval_count", self._count_tokens(prompt)),
            "response_tokens": result.get("eval_count", self._count_tokens(text)),
            "total_duration_ms": result.get("total_duration", 0) / 1_000_000,
            "eval_duration_ms": result.get("eval_duration", 0) / 1_000_000,
        }

    def list_models(self) -> list:
        try:
            url = self._base_url.rstrip("/") + "/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in result.get("models", [])]
        except Exception:
            return []