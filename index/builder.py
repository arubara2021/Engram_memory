from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from ..core.types import Chunk, Concept, IndexState

from .concept import ConceptExtractor
from .hash_table import MultiHashTable
from .ngram import NgramIndexer
from .graph import ConceptGraph


class IndexBuilder:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.concept_extractor = ConceptExtractor(config)
        self.ngram_indexer = NgramIndexer()

    def build(self, chunks: List[Chunk], full_text: str) -> IndexState:
        concepts = self.concept_extractor.extract(chunks, full_text)

        for concept in concepts:
            if not concept.primary_chunk_id and concept.chunk_ids:
                concept.primary_chunk_id = concept.chunk_ids[0]

        hash_tables = MultiHashTable()
        concepts_for_table = [
            {
                "term": c.label,
                "chunk_ids": c.chunk_ids,
                "score": c.score,
                "definition": c.definition,
                "primary_chunk_id": c.primary_chunk_id,
            }
            for c in concepts
        ]
        hash_tables.build_from_chunks(chunks, concepts_for_table)

        concept_graph = ConceptGraph()
        concepts_for_graph = [
            {
                "term": c.label,
                "chunk_ids": c.chunk_ids,
                "score": c.score,
            }
            for c in concepts
        ]
        concept_graph.build(concepts_for_graph, chunks)

        ngram_index = self.ngram_indexer.build_index(
            chunks, concepts, hash_tables
        )

        doc_id = chunks[0].doc_id if chunks else ""

        return IndexState(
            doc_id=doc_id,
            chunks=chunks,
            concepts=concepts,
            hash_tables=hash_tables,
            vector_store=None,
            concept_graph=concept_graph,
            ngram_index=ngram_index,
        )

    def update(
        self, new_chunks: List[Chunk], existing: IndexState, full_text: str
    ) -> IndexState:
        all_chunks = existing.chunks + new_chunks

        new_concepts = self.concept_extractor.extract(new_chunks, full_text)

        existing_labels = {c.label for c in existing.concepts}
        merged_concepts = list(existing.concepts)
        for nc in new_concepts:
            if nc.label in existing_labels:
                for ec in merged_concepts:
                    if ec.label == nc.label:
                        ec.frequency += nc.frequency
                        new_cids = set(nc.chunk_ids) - set(ec.chunk_ids)
                        ec.chunk_ids.extend(new_cids)
                        ec.chunk_ids = ec.chunk_ids[:15]
                        ec.score = max(ec.score, nc.score)
                        break
            else:
                if not nc.primary_chunk_id and nc.chunk_ids:
                    nc.primary_chunk_id = nc.chunk_ids[0]
                merged_concepts.append(nc)

        merged_concepts.sort(key=lambda c: c.score, reverse=True)
        max_c = getattr(self.config, "max_concepts", 80) if self.config else 80
        merged_concepts = merged_concepts[:max_c]

        hash_tables = MultiHashTable()
        concepts_for_table = [
            {
                "term": c.label,
                "chunk_ids": c.chunk_ids,
                "score": c.score,
                "definition": c.definition,
                "primary_chunk_id": c.primary_chunk_id,
            }
            for c in merged_concepts
        ]
        hash_tables.build_from_chunks(all_chunks, concepts_for_table)

        concept_graph = ConceptGraph()
        concepts_for_graph = [
            {
                "term": c.label,
                "chunk_ids": c.chunk_ids,
                "score": c.score,
            }
            for c in merged_concepts
        ]
        concept_graph.build(concepts_for_graph, all_chunks)

        ngram_index = self.ngram_indexer.build_index(
            all_chunks, merged_concepts, hash_tables
        )

        return IndexState(
            doc_id=existing.doc_id,
            chunks=all_chunks,
            concepts=merged_concepts,
            hash_tables=hash_tables,
            vector_store=existing.vector_store,
            concept_graph=concept_graph,
            ngram_index=ngram_index,
        )