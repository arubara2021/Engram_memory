from __future__ import annotations

import gzip
import json
import struct
import time
from typing import Any, Dict, List, Optional, Set

from ..core.types import Chunk, Concept, ConceptSlot, IndexState, SlotEntry


class Serializer:

    def serialize_index(self, state: IndexState) -> bytes:
        parts: List[bytes] = []

        parts.append(self._encode_string(state.doc_id))

        parts.append(struct.pack("<I", len(state.chunks)))
        for chunk in state.chunks:
            parts.append(self._encode_chunk(chunk))

        parts.append(struct.pack("<I", len(state.concepts)))
        for concept in state.concepts:
            parts.append(self._encode_concept(concept))

        if state.hash_tables is not None:
            parts.append(struct.pack("<B", 1))
            parts.append(self._encode_hash_tables(state.hash_tables))
        else:
            parts.append(struct.pack("<B", 0))

        if state.vector_store is not None:
            parts.append(struct.pack("<B", 1))
            parts.append(self._encode_vector_store(state.vector_store))
        else:
            parts.append(struct.pack("<B", 0))

        if state.concept_graph is not None:
            parts.append(struct.pack("<B", 1))
            parts.append(self._encode_concept_graph(state.concept_graph))
        else:
            parts.append(struct.pack("<B", 0))

        ngram_data = json.dumps(
            {k: list(v) for k, v in state.ngram_index.items()},
            ensure_ascii=False,
        ).encode("utf-8")
        parts.append(struct.pack("<I", len(ngram_data)))
        parts.append(ngram_data)

        raw = b"".join(parts)

        try:
            return gzip.compress(raw, compresslevel=6)
        except Exception:
            return raw

    def deserialize_index(self, data: bytes) -> IndexState:
        try:
            raw = gzip.decompress(data)
        except Exception:
            raw = data

        offset = 0

        def read(fmt: str, size: int):
            nonlocal offset
            val = struct.unpack_from(fmt, raw, offset)[0]
            offset += size
            return val

        def read_bytes() -> bytes:
            nonlocal offset
            length = read("<I", 4)
            val = raw[offset : offset + length]
            offset += length
            return val

        def read_string() -> str:
            return read_bytes().decode("utf-8")

        doc_id = read_string()

        num_chunks = read("<I", 4)
        chunks: List[Chunk] = []
        for _ in range(num_chunks):
            chunks.append(self._decode_chunk(raw, lambda: None, read, read_string))

        num_concepts = read("<I", 4)
        concepts: List[Concept] = []
        for _ in range(num_concepts):
            concepts.append(self._decode_concept(raw, read, read_string))

        hash_tables = None
        has_hash = read("<B", 1)
        if has_hash:
            hash_tables = self._decode_hash_tables(raw, read, read_string)

        vector_store = None
        has_vector = read("<B", 1)
        if has_vector:
            vector_store = self._decode_vector_store(raw, read, read_string)

        concept_graph = None
        has_graph = read("<B", 1)
        if has_graph:
            concept_graph = self._decode_concept_graph(raw, read, read_string)

        ngram_len = read("<I", 4)
        ngram_json = raw[offset : offset + ngram_len].decode("utf-8")
        offset += ngram_len
        ngram_data = json.loads(ngram_json)
        ngram_index = {k: set(v) for k, v in ngram_data.items()}

        return IndexState(
            doc_id=doc_id,
            chunks=chunks,
            concepts=concepts,
            hash_tables=hash_tables,
            vector_store=vector_store,
            concept_graph=concept_graph,
            ngram_index=ngram_index,
        )

    def _encode_string(self, text: str) -> bytes:
        encoded = text.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    def _encode_chunk(self, chunk: Chunk) -> bytes:
        parts: List[bytes] = []
        parts.append(self._encode_string(chunk.chunk_id))
        parts.append(self._encode_string(chunk.doc_id))
        parts.append(self._encode_string(chunk.text))
        parts.append(struct.pack("<I", chunk.word_count))
        parts.append(struct.pack("<I", chunk.page))
        parts.append(self._encode_string(chunk.section))
        parts.append(struct.pack("<I", chunk.index))

        has_emb = chunk.embedding is not None
        parts.append(struct.pack("<B", 1 if has_emb else 0))
        if has_emb and chunk.embedding:
            parts.append(struct.pack("<I", len(chunk.embedding)))
            for val in chunk.embedding:
                parts.append(struct.pack("<f", val))
        else:
            parts.append(struct.pack("<I", 0))

        parts.append(struct.pack("<I", len(chunk.concepts)))
        for c in chunk.concepts:
            parts.append(self._encode_string(c))

        parts.append(struct.pack("<I", len(chunk.words)))
        for w in chunk.words:
            parts.append(self._encode_string(w))

        parts.append(self._encode_string(chunk.original_text))

        return b"".join(parts)

    def _decode_chunk(
        self, data: bytes, _unused, read, read_string
    ) -> Chunk:
        chunk_id = read_string()
        doc_id = read_string()
        text = read_string()
        word_count = read("<I", 4)
        page = read("<I", 4)
        section = read_string()
        index = read("<I", 4)

        has_emb = read("<B", 1)
        emb_len = read("<I", 4)
        embedding = None
        if has_emb and emb_len > 0:
            embedding = []
            for _ in range(emb_len):
                embedding.append(read("<f", 4))

        num_concepts = read("<I", 4)
        concepts = [read_string() for _ in range(num_concepts)]

        num_words = read("<I", 4)
        words = [read_string() for _ in range(num_words)]

        original_text = read_string()

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            word_count=word_count,
            page=page,
            section=section,
            index=index,
            embedding=embedding,
            concepts=concepts,
            words=words,
            original_text=original_text,
        )

    def _encode_concept(self, concept: Concept) -> bytes:
        parts: List[bytes] = []
        parts.append(self._encode_string(concept.concept_id))
        parts.append(self._encode_string(concept.label))
        parts.append(struct.pack("<I", len(concept.ngrams)))
        for ng in concept.ngrams:
            parts.append(self._encode_string(ng))
        parts.append(struct.pack("<I", concept.frequency))
        parts.append(struct.pack("<I", len(concept.chunk_ids)))
        for cid in concept.chunk_ids:
            parts.append(self._encode_string(cid))
        parts.append(struct.pack("<f", concept.score))
        parts.append(struct.pack("<I", len(concept.tags)))
        for t in concept.tags:
            parts.append(self._encode_string(t))
        parts.append(self._encode_string(concept.primary_chunk_id or ""))
        parts.append(self._encode_string(concept.definition))
        parts.append(struct.pack("<B", 1 if concept.in_definition else 0))
        parts.append(struct.pack("<B", 1 if concept.in_heading else 0))
        parts.append(struct.pack("<B", 1 if concept.in_bold else 0))
        parts.append(struct.pack("<I", concept.word_count))
        return b"".join(parts)

    def _decode_concept(self, data: bytes, read, read_string) -> Concept:
        concept_id = read_string()
        label = read_string()
        num_ngrams = read("<I", 4)
        ngrams = [read_string() for _ in range(num_ngrams)]
        frequency = read("<I", 4)
        num_cids = read("<I", 4)
        chunk_ids = [read_string() for _ in range(num_cids)]
        score = read("<f", 4)
        num_tags = read("<I", 4)
        tags = [read_string() for _ in range(num_tags)]
        primary_chunk_id = read_string() or None
        definition = read_string()
        in_definition = bool(read("<B", 1))
        in_heading = bool(read("<B", 1))
        in_bold = bool(read("<B", 1))
        word_count = read("<I", 4)

        return Concept(
            concept_id=concept_id,
            label=label,
            ngrams=ngrams,
            frequency=frequency,
            chunk_ids=chunk_ids,
            score=score,
            tags=tags,
            primary_chunk_id=primary_chunk_id,
            definition=definition,
            in_definition=in_definition,
            in_heading=in_heading,
            in_bold=in_bold,
            word_count=word_count,
        )

    def _encode_hash_tables(self, ht: Any) -> bytes:
        if hasattr(ht, "serialize"):
            raw = ht.serialize()
            return struct.pack("<I", len(raw)) + raw
        return struct.pack("<I", 0)

    def _decode_hash_tables(self, data: bytes, read, read_string) -> Any:
        length = read("<I", 4)
        if length == 0:
            return None
        raw = data[
            struct.calcsize("<I")
            + (len(data) - len(data))
            :
        ]

        import importlib

        try:
            mod = importlib.import_module("engram.index.hash_table")
            cls = getattr(mod, "MultiHashTable")

            import struct as _struct

            current_offset = 0

            nonlocal_vars = {"offset": 0}

            def local_read(fmt: str, size: int):
                val = _struct.unpack_from(fmt, data, nonlocal_vars["offset"])[0]
                nonlocal_vars["offset"] += size
                return val

            def local_read_string() -> str:
                str_len = local_read("<I", 4)
                val = data[nonlocal_vars["offset"] : nonlocal_vars["offset"] + str_len].decode("utf-8")
                nonlocal_vars["offset"] += str_len
                return val

            nonlocal_vars["offset"] = read.__self__ if hasattr(read, "__self__") else 0

            return cls.deserialize(data[read.__self__ if hasattr(read, "__self__") else 0 :])

        except Exception:
            return None

    def _encode_vector_store(self, vs: Any) -> bytes:
        if hasattr(vs, "serialize"):
            raw = vs.serialize()
            return struct.pack("<I", len(raw)) + raw
        return struct.pack("<I", 0)

    def _decode_vector_store(self, data: bytes, read, read_string) -> Any:
        length = read("<I", 4)
        if length == 0:
            return None

        import importlib

        try:
            mod = importlib.import_module("engram.embed.store")
            cls = getattr(mod, "VectorStore")

            remaining = data[struct.calcsize("<I") : struct.calcsize("<I") + length]
            return cls.deserialize(remaining)
        except Exception:
            return None

    def _encode_concept_graph(self, cg: Any) -> bytes:
        if hasattr(cg, "serialize"):
            raw = cg.serialize()
            return struct.pack("<I", len(raw)) + raw
        return struct.pack("<I", 0)

    def _decode_concept_graph(self, data: bytes, read, read_string) -> Any:
        length = read("<I", 4)
        if length == 0:
            return None

        import importlib

        try:
            mod = importlib.import_module("engram.index.graph")
            cls = getattr(mod, "ConceptGraph")

            remaining = data[struct.calcsize("<I") : struct.calcsize("<I") + length]
            return cls.deserialize(remaining)
        except Exception:
            return None