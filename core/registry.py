from __future__ import annotations

from typing import Any, Dict, List, Type


class Registry:
    def __init__(self) -> None:
        self._parsers: Dict[str, Type] = {}
        self._chunkers: Dict[str, Type] = {}
        self._llms: Dict[str, Type] = {}
        self._parser_cache: Dict[str, Any] = {}
        self._chunker_cache: Dict[str, Any] = {}
        self._llm_cache: Dict[str, Any] = {}

    def register_parser(self, extensions: List[str], parser_class: Type) -> None:
        for ext in extensions:
            self._parsers[ext.lower().lstrip(".")] = parser_class

    def register_chunker(self, name: str, chunker_class: Type) -> None:
        self._chunkers[name.lower()] = chunker_class

    def register_llm(self, name: str, llm_class: Type) -> None:
        self._llms[name.lower()] = llm_class

    def get_parser(self, extension: str) -> Any:
        ext = extension.lower().lstrip(".")
        if ext not in self._parsers:
            raise ValueError(f"No parser registered for extension: .{ext}")
        if ext not in self._parser_cache:
            self._parser_cache[ext] = self._parsers[ext]()
        return self._parser_cache[ext]

    def get_chunker(self, name: str, config: Any = None) -> Any:
        key = name.lower()
        if key not in self._chunkers:
            raise ValueError(f"No chunker registered for strategy: {key}")
        cache_key = f"{key}_{id(config)}" if config else key
        if cache_key not in self._chunker_cache:
            if config is not None:
                self._chunker_cache[cache_key] = self._chunkers[key](config)
            else:
                self._chunker_cache[cache_key] = self._chunkers[key]()
        return self._chunker_cache[cache_key]

    def get_llm(self, name: str, config: Any = None) -> Any:
        key = name.lower()
        if key not in self._llms:
            raise ValueError(f"No LLM backend registered: {key}")
        cache_key = f"{key}_{id(config)}" if config else key
        if cache_key not in self._llm_cache:
            if config is not None:
                self._llm_cache[cache_key] = self._llms[key](config)
            else:
                self._llm_cache[cache_key] = self._llms[key]()
        return self._llm_cache[cache_key]

    def has_parser(self, extension: str) -> bool:
        return extension.lower().lstrip(".") in self._parsers

    def has_chunker(self, name: str) -> bool:
        return name.lower() in self._chunkers

    def has_llm(self, name: str) -> bool:
        return name.lower() in self._llms

    def list_parsers(self) -> List[str]:
        return sorted(set(self._parsers.keys()))

    def list_chunkers(self) -> List[str]:
        return sorted(self._chunkers.keys())

    def list_llms(self) -> List[str]:
        return sorted(self._llms.keys())

    def list_parser_classes(self) -> List[str]:
        return sorted(set(c.__name__ for c in self._parsers.values()))

    def clear_cache(self) -> None:
        self._parser_cache.clear()
        self._chunker_cache.clear()
        self._llm_cache.clear()