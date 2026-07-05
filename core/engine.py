from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from ..core.config import EngramConfig
from ..core.types import Chunk, ChunkResult, DocumentState


def document_relevance(query: str, chunks: List[Chunk], concepts: list) -> float:
    words = query.lower().split()
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "with", "and", "or", "but", "how", "what",
        "why", "when", "where", "do", "does", "did", "can", "could",
        "would", "should", "this", "that", "it", "its", "be", "been",
        "being", "have", "has", "had", "about", "up", "down", "i", "me",
        "my", "we", "our", "you", "your", "he", "she", "they", "them",
    }
    content_words = {w for w in words if w not in stop_words and len(w) > 2}
    if not content_words:
        return 1.0

    doc_text = " ".join(c.text[:200] for c in chunks[:30]).lower()
    concept_labels = {c.label.lower() for c in concepts}

    hits = 0
    for cw in content_words:
        if cw in doc_text:
            hits += 1
        else:
            for cl in concept_labels:
                if cw in cl:
                    hits += 0.5
                    break

    return hits / len(content_words)


class Engram:

    def __init__(self, config: Optional[EngramConfig] = None) -> None:
        self.config = config or EngramConfig()
        self.documents: Dict[str, DocumentState] = {}
        self._doc_counter = 0

    def ingest(self, filepath: str) -> str:
        import time
        start = time.perf_counter()
        self._doc_counter += 1
        doc_id = f"{self._doc_counter:08x}"

        from ..ingest.pdf import PDFParser
        parser = PDFParser()
        parsed = parser.parse(filepath)

        from ..chunk.factory import ChunkerFactory
        strategy = getattr(self.config, "chunk_strategy", "recursive")
        chunker = ChunkerFactory.create(strategy, self.config)
        metadata = {"source": filepath, "title": parsed.metadata.get("title", "")}
        chunks = chunker.chunk(parsed.full_text, doc_id, metadata, parsed.page_spans)

        from ..chunk.overlap import OverlapHandler
        overlap = OverlapHandler()
        chunks = overlap.apply(chunks)

        from ..index.concept import ConceptExtractor
        extractor = ConceptExtractor(self.config)
        concepts = extractor.extract(chunks, parsed.full_text)

        from ..index.hash_table import MultiHashTable
        hash_tables = MultiHashTable()
        concept_dicts = [
            {"term": c.label, "chunk_ids": c.chunk_ids, "definition": c.definition, "primary_chunk_id": c.primary_chunk_id}
            for c in concepts
        ]
        hash_tables.build_from_chunks(chunks, concept_dicts)

        from ..core.types import Document, IndexState
        document = Document(
            doc_id=doc_id,
            title=parsed.metadata.get("title", filepath),
            source_path=filepath,
            full_text=parsed.full_text,
            pages=parsed.page_count,
            total_chars=parsed.total_chars,
            total_words=parsed.total_words,
            text_quality=parsed.text_quality,
            domain="unknown",
            language="en",
        )

        index = IndexState(
            doc_id=doc_id,
            chunks=chunks,
            concepts=concepts,
            hash_tables=hash_tables,
        )

        state = DocumentState(document=document, index=index)
        self.documents[doc_id] = state
        return doc_id

    def ingest_text(self, text: str, source: str = "text") -> str:
        import time
        start = time.perf_counter()
        self._doc_counter += 1
        doc_id = f"{self._doc_counter:08x}"

        words = text.split()
        quality = 1.0 - min(len(text) * 0.00001, 0.5)

        from ..chunk.factory import ChunkerFactory
        strategy = getattr(self.config, "chunk_strategy", "recursive")
        chunker = ChunkerFactory.create(strategy, self.config)
        metadata = {"source": source, "title": source}
        chunks = chunker.chunk(text, doc_id, metadata, [(1, len(text), 1)])

        from ..chunk.overlap import OverlapHandler
        overlap = OverlapHandler()
        chunks = overlap.apply(chunks)

        from ..index.concept import ConceptExtractor
        extractor = ConceptExtractor(self.config)
        concepts = extractor.extract(chunks, text)

        from ..index.hash_table import MultiHashTable
        hash_tables = MultiHashTable()
        concept_dicts = [
            {"term": c.label, "chunk_ids": c.chunk_ids, "definition": c.definition, "primary_chunk_id": c.primary_chunk_id}
            for c in concepts
        ]
        hash_tables.build_from_chunks(chunks, concept_dicts)

        from ..core.types import Document, IndexState
        document = Document(
            doc_id=doc_id,
            title=source,
            source_path=source,
            full_text=text,
            pages=1,
            total_chars=len(text),
            total_words=len(words),
            text_quality=quality,
            domain="unknown",
            language="en",
        )

        index = IndexState(
            doc_id=doc_id,
            chunks=chunks,
            concepts=concepts,
            hash_tables=hash_tables,
        )

        state = DocumentState(document=document, index=index)
        self.documents[doc_id] = state
        return doc_id

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[ChunkResult]:
        if not self.documents:
            raise RuntimeError(
                "No documents ingested. Call ingest() first."
            )

        effective_top_k = top_k or self.config.final_top_k
        effective_method = method or self.config.retrieval_method

        if doc_id:
            if doc_id not in self.documents:
                raise ValueError(f"Document not found: {doc_id}")
            states = [self.documents[doc_id]]
        else:
            states = list(self.documents.values())

        if not doc_id and len(states) > 1:
            relevant_states = []
            for state in states:
                relevance = document_relevance(
                    query,
                    state.index.chunks,
                    state.index.concepts,
                )
                if relevance >= 0.15:
                    relevant_states.append(state)
            if relevant_states:
                states = relevant_states

        all_results: List[ChunkResult] = []

        for state in states:
            if effective_method in ("hash", "dual"):
                hash_results = self._hash_retrieve(
                    query, state.index, effective_top_k * 3
                )
                all_results.extend(hash_results)

            if (
                effective_method in ("vector", "dual")
                and self.config.embedding_enabled
            ):
                vector_results = self._vector_retrieve(
                    query, state.index, effective_top_k * 3
                )
                all_results.extend(vector_results)

        if effective_method == "dual" and self.config.embedding_enabled:
            all_results = self._merge_dual(all_results)

        all_results = self._rerank(all_results, query)
        all_results = self._apply_gate(all_results, query=query)

        seen: set = set()
        deduped: List[ChunkResult] = []
        for r in all_results:
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                deduped.append(r)

        return deduped[:effective_top_k]

    def _hash_retrieve(
        self, query: str, index: Any, top_k: int
    ) -> List[ChunkResult]:
        if index.hash_tables is None:
            return []

        try:
            mod = importlib.import_module("engram.retrieve.hash_retriever")
            retriever = mod.HashRetriever(self.config)
            return retriever.retrieve(
                query, index.hash_tables, index.chunks, top_k,
                doc_id=index.doc_id,
            )
        except ImportError:
            pass

        results = index.hash_tables.lookup_with_expansion(
            {"term": query, "chunk_ids": [], "definition": ""},
            index.chunks,
        )
        chunk_map = {c.chunk_id: c for c in index.chunks}
        output: List[ChunkResult] = []
        for entry in results:
            chunk = chunk_map.get(entry.chunk_id)
            if chunk:
                output.append(
                    ChunkResult(
                        chunk=chunk,
                        score=entry.relevance_score,
                        source="hash",
                        hash_hits=1,
                    )
                )
        return output[:top_k]

    def _vector_retrieve(
        self, query: str, index: Any, top_k: int
    ) -> List[ChunkResult]:
        if index.embeddings is None:
            return []

        try:
            mod = importlib.import_module("engram.retrieve.dense_retriever")
            retriever = mod.DenseRetriever(self.config)
            return retriever.retrieve(
                query, index.chunks, index.embeddings, top_k
            )
        except ImportError:
            return []

    def _merge_dual(
        self, results: List[ChunkResult]
    ) -> List[ChunkResult]:
        chunk_scores: Dict[str, ChunkResult] = {}
        for r in results:
            cid = r.chunk.chunk_id
            if cid in chunk_scores:
                existing = chunk_scores[cid]
                if r.source == "vector":
                    existing.vector_sim = r.vector_sim
                    existing.score = (existing.score + r.vector_sim) / 2
                else:
                    existing.hash_hits = max(existing.hash_hits, r.hash_hits)
                    existing.score = max(existing.score, r.score)
            else:
                chunk_scores[cid] = r
        return list(chunk_scores.values())

    def _rerank(
        self, results: List[ChunkResult], query: str
    ) -> List[ChunkResult]:
        if not results:
            return []

        try:
            mod = importlib.import_module("engram.retrieve.reranker")
            reranker = mod.ReRanker(self.config)
            chunks = list({r.chunk for r in results})
            return reranker.rerank(results, query, chunks=chunks)
        except ImportError:
            pass

        results.sort(key=lambda r: -r.score)
        return results

    def _apply_gate(
        self, results: List[ChunkResult], query: str = ""
    ) -> List[ChunkResult]:
        if not self.config.context_gate_enabled:
            return results

        try:
            mod = importlib.import_module("engram.retrieve.context_gate")
            gate = mod.ContextGate(self.config)
            return gate.decide(results, query=query)
        except ImportError:
            pass

        filtered = [
            r
            for r in results
            if r.score >= self.config.gate_relevance_threshold
        ]
        if filtered:
            return filtered[: self.config.gate_max_chunks]
        return results[: self.config.gate_min_chunks]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        results = self.retrieve(question, top_k, method, doc_id)
        if not results:
            return "No relevant information found."

        answer_parts = []
        for i, r in enumerate(results[:3], 1):
            answer_parts.append(f"[{i}] {r.chunk.text[:200]}")

        return "\n\n".join(answer_parts)

    def get_stats(self) -> Dict[str, Any]:
        total_chunks = sum(
            len(s.index.chunks) for s in self.documents.values()
        )
        total_concepts = sum(
            len(s.index.concepts) for s in self.documents.values()
        )
        total_words = sum(
            s.document.total_words for s in self.documents.values()
        )
        total_hash = sum(
            s.index.hash_tables.total_insertions
            for s in self.documents.values()
            if s.index.hash_tables
        )

        return {
            "documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_concepts": total_concepts,
            "total_words": total_words,
            "total_hash_entries": total_hash,
        }

    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        state = self.documents[doc_id]
        return {
            "doc_id": doc_id,
            "filepath": state.document.source_path,
            "total_words": state.document.total_words,
            "total_chars": state.document.total_chars,
            "text_quality": state.document.text_quality,
            "chunks": len(state.index.chunks),
            "concepts": len(state.index.concepts),
            "hash_entries": (
                state.index.hash_tables.total_insertions
                if state.index.hash_tables
                else 0
            ),
        }

    def get_concepts(
        self, doc_id: str, top_n: int = 20
    ) -> List[Dict[str, Any]]:
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        state = self.documents[doc_id]
        sorted_concepts = sorted(
            state.index.concepts, key=lambda c: c.score, reverse=True
        )
        return [
            {
                "label": c.label,
                "score": c.score,
                "frequency": c.frequency,
                "tags": c.tags,
            }
            for c in sorted_concepts[:top_n]
        ]

    def get_context(
        self, doc_id: str, chunk_index: int, window: int = 1
    ) -> List[str]:
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        state = self.documents[doc_id]
        chunks = state.index.chunks
        start = max(0, chunk_index - window)
        end = min(len(chunks), chunk_index + window + 1)
        return [c.text for c in chunks[start:end]]
