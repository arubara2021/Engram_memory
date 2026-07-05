from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional


class LRUCache:

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def has(self, key: str) -> bool:
        return key in self._cache

    def remove(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def peek(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def keys(self) -> list:
        return list(self._cache.keys())

    def values(self) -> list:
        return list(self._cache.values())

    def items(self) -> list:
        return list(self._cache.items())

    def size(self) -> int:
        return len(self._cache)

    def is_empty(self) -> bool:
        return len(self._cache) == 0

    def is_full(self) -> bool:
        return len(self._cache) >= self.capacity

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total * 100) if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate(), 2),
        }

    def resize(self, new_capacity: int) -> None:
        self.capacity = max(1, new_capacity)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def get_or_compute(self, key: str, compute_fn: Any) -> Any:
        value = self.get(key)
        if value is not None:
            return value
        value = compute_fn(key)
        self.put(key, value)
        return value

    def update(self, other: Any) -> None:
        if isinstance(other, dict):
            for k, v in other.items():
                self.put(k, v)
        elif isinstance(other, LRUCache):
            for k, v in other.items():
                self.put(k, v)