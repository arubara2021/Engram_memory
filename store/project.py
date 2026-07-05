from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.types import IndexState

from .filesystem import FileStore


class ProjectStore:

    def __init__(self, base_path: str = "./engram_data") -> None:
        self.base_path = os.path.abspath(base_path)
        self._file_store = FileStore(base_path)
        self._projects_dir = os.path.join(self.base_path, "_projects")

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _project_dir(self, project_id: str) -> str:
        return os.path.join(self._projects_dir, project_id)

    def _project_meta_path(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "project.json")

    def create_project(
        self,
        name: str,
        description: str = "",
        project_id: Optional[str] = None,
    ) -> str:
        import hashlib
        import uuid

        if not project_id:
            project_id = hashlib.md5(
                f"{name}_{uuid.uuid4()}".encode()
            ).hexdigest()[:12]

        proj_dir = self._project_dir(project_id)
        self._ensure_dir(proj_dir)

        meta = {
            "project_id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "documents": [],
        }

        meta_path = self._project_meta_path(project_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return project_id

    def add_document(
        self,
        project_id: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = self._load_project_meta(project_id)
        if meta is None:
            raise ValueError(f"Project not found: {project_id}")

        doc_entry = {
            "doc_id": doc_id,
            "added_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        existing_ids = {d["doc_id"] for d in meta["documents"]}
        if doc_id not in existing_ids:
            meta["documents"].append(doc_entry)

        meta["updated_at"] = datetime.utcnow().isoformat()
        self._save_project_meta(project_id, meta)

    def remove_document(self, project_id: str, doc_id: str) -> bool:
        meta = self._load_project_meta(project_id)
        if meta is None:
            return False

        original_count = len(meta["documents"])
        meta["documents"] = [
            d for d in meta["documents"] if d["doc_id"] != doc_id
        ]

        if len(meta["documents"]) < original_count:
            meta["updated_at"] = datetime.utcnow().isoformat()
            self._save_project_meta(project_id, meta)
            return True

        return False

    def list_documents(self, project_id: str) -> List[Dict[str, Any]]:
        meta = self._load_project_meta(project_id)
        if meta is None:
            return []
        return meta.get("documents", [])

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self._load_project_meta(project_id)

    def list_projects(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._projects_dir):
            return []

        projects: List[Dict[str, Any]] = []
        for entry in os.listdir(self._projects_dir):
            entry_path = os.path.join(self._projects_dir, entry)
            if os.path.isdir(entry_path):
                meta = self._load_project_meta(entry)
                if meta:
                    projects.append(meta)

        projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        return projects

    def delete_project(
        self, project_id: str, delete_documents: bool = False
    ) -> bool:
        meta = self._load_project_meta(project_id)
        if meta is None:
            return False

        if delete_documents:
            for doc_entry in meta.get("documents", []):
                doc_id = doc_entry["doc_id"]
                self._file_store.delete(doc_id)

        proj_dir = self._project_dir(project_id)
        import shutil

        if os.path.exists(proj_dir):
            shutil.rmtree(proj_dir)
            return True

        return False

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        meta = self._load_project_meta(project_id)
        if meta is None:
            return False

        if name is not None:
            meta["name"] = name
        if description is not None:
            meta["description"] = description

        meta["updated_at"] = datetime.utcnow().isoformat()
        self._save_project_meta(project_id, meta)
        return True

    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        meta = self._load_project_meta(project_id)
        if meta is None:
            return {}

        doc_ids = [d["doc_id"] for d in meta.get("documents", [])]
        total_chunks = 0
        total_concepts = 0
        total_size = 0

        for doc_id in doc_ids:
            file_meta = self._file_store.get_metadata(doc_id)
            if file_meta:
                total_chunks += file_meta.get("chunk_count", 0)
                total_concepts += file_meta.get("concept_count", 0)
                total_size += file_meta.get("index_size_bytes", 0)

        return {
            "project_id": project_id,
            "name": meta.get("name", ""),
            "document_count": len(doc_ids),
            "total_chunks": total_chunks,
            "total_concepts": total_concepts,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
        }

    def _load_project_meta(self, project_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._project_meta_path(project_id)
        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_project_meta(
        self, project_id: str, meta: Dict[str, Any]
    ) -> None:
        proj_dir = self._project_dir(project_id)
        self._ensure_dir(proj_dir)

        meta_path = self._project_meta_path(project_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)