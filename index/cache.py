from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class ApproximateCache:

    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.85) -> None:
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self._cache: OrderedDict = OrderedDict()
        self._signatures: Dict[str, str] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._fingerprints: Dict[str, List[int]] = {}

    def get(self, query: str) -> Optional[Any]:
        key = self._hash_key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        similar = self._find_similar(query)
        if similar is not None:
            self._hits += 1
            return similar

        self._misses += 1
        return None

    def put(self, query: str, results: Any) -> None:
        key = self._hash_key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._fingerprints.pop(evicted_key, None)
            self._signatures.pop(evicted_key, None)

        self._cache[key] = results
        self._signatures[key] = query.lower().strip()
        self._fingerprints[key] = self._compute_fingerprint(query)

    def has(self, query: str) -> bool:
        key = self._hash_key(query)
        if key in self._cache:
            return True
        return self._find_similar(query) is not None

    def _find_similar(self, query: str) -> Optional[Any]:
        if not self._fingerprints:
            return None

        fp = self._compute_fingerprint(query)
        best_key = None
        best_sim = 0.0

        for key, cached_fp in self._fingerprints.items():
            sim = self._fingerprint_similarity(fp, cached_fp)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None and best_key in self._cache:
            self._cache.move_to_end(best_key)
            return self._cache[best_key]

        return None

    def _compute_fingerprint(self, text: str) -> List[int]:
        words = text.lower().split()
        words = [w.strip(".,;:!?\"'()[]{}") for w in words if len(w) > 1]
        if not words:
            return [0] * 16

        fingerprint = [0] * 16
        for i, word in enumerate(words):
            h = self._simple_hash(word)
            idx = h % 16
            fingerprint[idx] += 1
            if i + 1 < len(words):
                bigram = f"{words[i]}_{words[i+1]}"
                h2 = self._simple_hash(bigram)
                idx2 = h2 % 16
                fingerprint[idx2] += 2

        return fingerprint

    def _fingerprint_similarity(self, a: List[int], b: List[int]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _simple_hash(self, text: str) -> int:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    def _hash_key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        self._cache.clear()
        self._signatures.clear()
        self._fingerprints.clear()
        self._hits = 0
        self._misses = 0

    def size(self) -> int:
        return len(self._cache)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total * 100) if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate(), 2),
            "similarity_threshold": self.similarity_threshold,
        }