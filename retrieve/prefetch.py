from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional


class LRUCache:

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def has(self, key: str) -> bool:
        return key in self._cache

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "capacity": self.max_size,
            "size": self.size(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate(), 2),
        }


class Prefetcher:

    def __init__(self, cache_size: int = 10, predict_count: int = 3) -> None:
        self._cache = LRUCache(max_size=cache_size)
        self._predict_count = predict_count
        self._access_history: List[str] = []
        self._max_history = 50

    def predict_next(
        self,
        concepts: List[Dict],
        count: Optional[int] = None,
        current_idx: int = 0,
    ) -> List[str]:
        n = count or self._predict_count
        remaining = concepts[current_idx : current_idx + 20]
        if not remaining:
            remaining = concepts[:20]

        sorted_c = sorted(remaining, key=lambda c: c.get("score", 0), reverse=True)
        return [c["term"] for c in sorted_c[:n]]

    def predict_from_history(
        self,
        concepts: List[Dict],
        concept_graph: Any = None,
        count: Optional[int] = None,
    ) -> List[str]:
        n = count or self._predict_count
        candidates: Dict[str, float] = {}

        for concept in concepts:
            term = concept.get("term", "")
            if term in candidates:
                continue
            score = concept.get("score", 0)
            candidates[term] = score * 0.5

        if concept_graph and self._access_history:
            for recent in self._access_history[-5:]:
                related = concept_graph.get_related(recent, max_depth=1)
                for r in related:
                    if r not in candidates:
                        candidates[r] = 0.8
                    else:
                        candidates[r] += 0.3

        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return [term for term, _ in ranked[:n]]

    def prefetch(self, concept_ids: List[str], retriever_fn: Any = None) -> None:
        if retriever_fn is None:
            return
        for cid in concept_ids:
            if not self._cache.has(cid):
                try:
                    results = retriever_fn(cid)
                    self._cache.put(cid, results)
                except Exception:
                    pass

    def get(self, concept_id: str) -> Optional[Any]:
        return self._cache.get(concept_id)

    def put(self, key: str, value: Any) -> None:
        self._cache.put(key, value)

    def has(self, concept_id: str) -> bool:
        return self._cache.has(concept_id)

    def record_access(self, concept_id: str) -> None:
        self._access_history.append(concept_id)
        if len(self._access_history) > self._max_history:
            self._access_history = self._access_history[-self._max_history :]

    def hit_rate(self) -> float:
        return self._cache.hit_rate()

    def clear(self) -> None:
        self._cache.clear()
        self._access_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hit_rate": round(self._cache.hit_rate(), 2),
            "cache_size": self._cache.size(),
            "total_hits": self._cache.hits,
            "total_misses": self._cache.misses,
            "history_length": len(self._access_history),
        }