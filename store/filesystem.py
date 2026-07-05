from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.types import IndexState

from .serializer import Serializer


class FileStore:

    def __init__(self, base_path: str = "./engram_data") -> None:
        self.base_path = os.path.abspath(base_path)
        self._serializer = Serializer()

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _doc_dir(self, doc_id: str) -> str:
        return os.path.join(self.base_path, doc_id)

    def _index_path(self, doc_id: str) -> str:
        return os.path.join(self._doc_dir(doc_id), "engram.bin")

    def _meta_path(self, doc_id: str) -> str:
        return os.path.join(self._doc_dir(doc_id), "meta.json")

    def save(
        self,
        doc_id: str,
        state: IndexState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        doc_dir = self._doc_dir(doc_id)
        self._ensure_dir(doc_dir)

        serialized = self._serializer.serialize_index(state)

        index_path = self._index_path(doc_id)
        with open(index_path, "wb") as f:
            f.write(serialized)

        meta = {
            "doc_id": doc_id,
            "saved_at": datetime.utcnow().isoformat(),
            "index_size_bytes": len(serialized),
            "chunk_count": len(state.chunks),
            "concept_count": len(state.concepts),
            "has_hash_tables": state.hash_tables is not None,
            "has_vector_store": state.vector_store is not None,
            "has_concept_graph": state.concept_graph is not None,
            "ngram_count": len(state.ngram_index),
        }
        if metadata:
            meta.update(metadata)

        meta_path = self._meta_path(doc_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def load(self, doc_id: str) -> IndexState:
        index_path = self._index_path(doc_id)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found for document: {doc_id}")

        with open(index_path, "rb") as f:
            data = f.read()

        return self._serializer.deserialize_index(data)

    def delete(self, doc_id: str) -> bool:
        doc_dir = self._doc_dir(doc_id)
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
            return True
        return False

    def exists(self, doc_id: str) -> bool:
        index_path = self._index_path(doc_id)
        return os.path.exists(index_path)

    def list_documents(self) -> List[str]:
        if not os.path.exists(self.base_path):
            return []

        doc_ids: List[str] = []
        for entry in os.listdir(self.base_path):
            entry_path = os.path.join(self.base_path, entry)
            if os.path.isdir(entry_path):
                index_path = os.path.join(entry_path, "engram.bin")
                if os.path.exists(index_path):
                    doc_ids.append(entry)

        return sorted(doc_ids)

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._meta_path(doc_id)
        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for doc_id in self.list_documents():
            meta = self.get_metadata(doc_id)
            if meta:
                results.append(meta)
        return results

    def get_storage_size(self) -> Dict[str, Any]:
        total_bytes = 0
        file_count = 0
        doc_sizes: Dict[str, int] = {}

        if not os.path.exists(self.base_path):
            return {
                "total_bytes": 0,
                "total_mb": 0.0,
                "file_count": 0,
                "document_count": 0,
                "doc_sizes": {},
            }

        for doc_id in self.list_documents():
            doc_dir = self._doc_dir(doc_id)
            doc_size = 0
            for root, dirs, files in os.walk(doc_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                        doc_size += size
                        total_bytes += size
                        file_count += 1
                    except OSError:
                        pass
            doc_sizes[doc_id] = doc_size

        return {
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "file_count": file_count,
            "document_count": len(doc_sizes),
            "doc_sizes": doc_sizes,
        }

    def clear(self) -> int:
        count = 0
        for doc_id in self.list_documents():
            if self.delete(doc_id):
                count += 1
        return count

    def backup(self, backup_path: str) -> str:
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Storage path not found: {self.base_path}")

        backup_dir = os.path.abspath(backup_path)
        shutil.copytree(self.base_path, backup_dir, dirs_exist_ok=True)
        return backup_dir

    def restore(self, backup_path: str) -> int:
        backup_dir = os.path.abspath(backup_path)
        if not os.path.exists(backup_dir):
            raise FileNotFoundError(f"Backup not found: {backup_dir}")

        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)

        shutil.copytree(backup_dir, self.base_path)
        return len(self.list_documents())