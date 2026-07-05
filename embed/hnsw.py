from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class HNSWStore:

    def __init__(
        self,
        dimension: int = 384,
        max_elements: int = 100000,
        ef_construction: int = 200,
        m: int = 16,
    ) -> None:
        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = m
        self._index: Any = None
        self._id_to_label: Dict[str, int] = {}
        self._label_to_id: Dict[int, str] = {}
        self._next_label: int = 0
        self._is_built: bool = False

    def is_available(self) -> bool:
        try:
            import hnswlib

            return True
        except ImportError:
            return False

    def build(self, vectors: List[List[float]], ids: List[str]) -> None:
        if not vectors or not ids:
            return

        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is required for HNSW indexing. "
                "Install with: pip install hnswlib"
            )

        self._index = hnswlib.Index(space="cosine", dim=self.dimension)
        self._index.init_index(
            max_elements=max(self.max_elements, len(vectors) * 2),
            ef_construction=self.ef_construction,
            M=self.m,
        )

        self._id_to_label = {}
        self._label_to_id = {}
        self._next_label = 0

        import numpy as np

        data = []
        labels = []
        for vec, cid in zip(vectors, ids):
            label = self._next_label
            self._id_to_label[cid] = label
            self._label_to_id[label] = cid
            self._next_label += 1
            data.append(vec)
            labels.append(label)

        self._index.add_items(np.array(data, dtype=np.float32), labels)
        self._index.set_ef(50)
        self._is_built = True

    def add(self, vector: List[float], chunk_id: str) -> None:
        if not self._is_built:
            raise RuntimeError("HNSW index not built. Call build() first.")

        try:
            import numpy as np

            label = self._next_label
            self._id_to_label[chunk_id] = label
            self._label_to_id[label] = chunk_id
            self._next_label += 1

            self._index.add_items(np.array([vector], dtype=np.float32), [label])
        except Exception:
            pass

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if not self._is_built or self._index is None:
            return []

        try:
            import numpy as np

            labels, distances = self._index.knn_query(
                np.array([query_vector], dtype=np.float32), k=top_k
            )

            results: List[Tuple[str, float]] = []
            for label, dist in zip(labels[0], distances[0]):
                chunk_id = self._label_to_id.get(int(label))
                if chunk_id is not None:
                    similarity = 1.0 - float(dist)
                    results.append((chunk_id, similarity))

            return results
        except Exception:
            return []

    def remove(self, chunk_id: str) -> None:
        label = self._id_to_label.pop(chunk_id, None)
        if label is not None:
            self._label_to_id.pop(label, None)
            if self._index is not None:
                try:
                    self._index.mark_deleted(label)
                except Exception:
                    pass

    def size(self) -> int:
        return len(self._id_to_label)

    def clear(self) -> None:
        self._index = None
        self._id_to_label.clear()
        self._label_to_id.clear()
        self._next_label = 0
        self._is_built = False