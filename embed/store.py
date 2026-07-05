from __future__ import annotations

import json
import math
import struct
from typing import Any, Dict, List, Optional, Tuple


class VectorStore:

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._norms: Dict[str, float] = {}

    def add(self, chunk_id: str, vector: List[float], metadata: Optional[Dict] = None) -> None:
        self._vectors[chunk_id] = vector
        self._metadata[chunk_id] = metadata or {}
        self._norms[chunk_id] = math.sqrt(sum(v * v for v in vector))

    def add_batch(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        for i, (cid, vec) in enumerate(zip(chunk_ids, vectors)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self.add(cid, vec, meta)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        if not self._vectors:
            return []

        query_norm = math.sqrt(sum(v * v for v in query_vector))
        if query_norm == 0:
            return []

        results: List[Tuple[str, float]] = []
        query_normalized = [v / query_norm for v in query_vector]

        for chunk_id, vector in self._vectors.items():
            vec_norm = self._norms.get(chunk_id, 0.0)
            if vec_norm == 0:
                continue

            dot = sum(a * b for a, b in zip(query_normalized, vector))
            similarity = dot / vec_norm if vec_norm > 0 else 0.0

            if similarity >= threshold:
                results.append((chunk_id, similarity))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def remove(self, chunk_id: str) -> None:
        self._vectors.pop(chunk_id, None)
        self._metadata.pop(chunk_id, None)
        self._norms.pop(chunk_id, None)

    def get(self, chunk_id: str) -> Optional[List[float]]:
        return self._vectors.get(chunk_id)

    def get_metadata(self, chunk_id: str) -> Optional[Dict]:
        return self._metadata.get(chunk_id)

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._vectors

    def size(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()
        self._metadata.clear()
        self._norms.clear()

    def get_all_ids(self) -> List[str]:
        return list(self._vectors.keys())

    def serialize(self) -> bytes:
        data = {
            "dimension": self.dimension,
            "vectors": self._vectors,
            "metadata": {
                k: {mk: mv for mk, mv in v.items() if isinstance(mv, (str, int, float, bool, type(None)))}
                for k, v in self._metadata.items()
            },
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> VectorStore:
        parsed = json.loads(data.decode("utf-8"))
        store = cls(dimension=parsed.get("dimension", 384))
        for chunk_id, vector in parsed.get("vectors", {}).items():
            meta = parsed.get("metadata", {}).get(chunk_id, {})
            store.add(chunk_id, vector, meta)
        return store

    def serialize_binary(self) -> bytes:
        parts: List[bytes] = []
        parts.append(struct.pack("<I", self.dimension))
        parts.append(struct.pack("<I", len(self._vectors)))

        for chunk_id, vector in self._vectors.items():
            cid_bytes = chunk_id.encode("utf-8")
            parts.append(struct.pack("<H", len(cid_bytes)))
            parts.append(cid_bytes)
            parts.append(struct.pack("<I", len(vector)))
            for val in vector:
                parts.append(struct.pack("<f", val))

        return b"".join(parts)

    @classmethod
    def deserialize_binary(cls, data: bytes) -> VectorStore:
        offset = 0

        def read(fmt: str, size: int):
            nonlocal offset
            val = struct.unpack_from(fmt, data, offset)[0]
            offset += size
            return val

        dimension = read("<I", 4)
        store = cls(dimension=dimension)
        num_vectors = read("<I", 4)

        for _ in range(num_vectors):
            cid_len = read("<H", 2)
            chunk_id = data[offset : offset + cid_len].decode("utf-8")
            offset += cid_len
            vec_len = read("<I", 4)
            vector = []
            for _ in range(vec_len):
                val = read("<f", 4)
                vector.append(val)
            store.add(chunk_id, vector)

        return store