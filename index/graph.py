from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.types import Chunk, Concept, ConceptEdge


class ConceptGraph:

    def __init__(self) -> None:
        self.nodes: Dict[str, Concept] = {}
        self.edges: List[ConceptEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._edge_weights: Dict[Tuple[str, str], float] = {}

    def build(self, concepts: List[Dict], chunks: List[Chunk]) -> None:
        concept_map: Dict[str, List[str]] = {}
        for c in concepts:
            concept_map[c["term"]] = c.get("chunk_ids", [])

        chunk_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        for term, cids in concept_map.items():
            for cid in cids:
                chunk_to_concepts[cid].add(term)

        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        for cid, terms in chunk_to_concepts.items():
            term_list = sorted(terms)
            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    a, b = term_list[i], term_list[j]
                    key = (a, b)
                    cooccurrence[key] += 1

        for (a, b), count in cooccurrence.items():
            if count >= 1:
                weight = min(count / 5.0, 1.0)
                edge = ConceptEdge(
                    source_id=a,
                    target_id=b,
                    weight=weight,
                    edge_type="co_occurrence",
                )
                self.edges.append(edge)
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)
                self._edge_weights[(a, b)] = weight
                self._edge_weights[(b, a)] = weight

        hier_edges = self._detect_hierarchical(concept_map)
        for edge in hier_edges:
            self.edges.append(edge)
            self.adjacency[edge.source_id].add(edge.target_id)
            self.adjacency[edge.target_id].add(edge.source_id)
            self._edge_weights[(edge.source_id, edge.target_id)] = edge.weight
            self._edge_weights[(edge.target_id, edge.source_id)] = edge.weight

    def _detect_hierarchical(
        self, concept_map: Dict[str, List[str]]
    ) -> List[ConceptEdge]:
        edges: List[ConceptEdge] = []
        items = list(concept_map.items())

        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                term_a, cids_a = items[i]
                term_b, cids_b = items[j]
                set_a = set(cids_a)
                set_b = set(cids_b)

                if set_b and set_a.issuperset(set_b) and len(set_a) > len(set_b):
                    if term_a.split() and term_b.split():
                        a_words = set(term_a.lower().split())
                        b_words = set(term_b.lower().split())
                        if a_words.issubset(b_words) or b_words.issubset(a_words):
                            more_general = term_a if len(set_a) > len(set_b) else term_b
                            more_specific = term_b if more_general == term_a else term_a
                            edges.append(
                                ConceptEdge(
                                    source_id=more_general,
                                    target_id=more_specific,
                                    weight=0.8,
                                    edge_type="hierarchical",
                                )
                            )

        return edges

    def get_related(self, concept_id: str, max_depth: int = 2) -> List[str]:
        visited: Set[str] = set()
        queue = [(concept_id, 0)]
        result: List[str] = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            if current != concept_id:
                result.append(current)
            if depth < max_depth:
                for neighbor in self.adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return result

    def get_prerequisites(self, concept_id: str) -> List[str]:
        prereqs: List[str] = []
        for edge in self.edges:
            if edge.target_id == concept_id and edge.edge_type == "hierarchical":
                prereqs.append(edge.source_id)
        return prereqs

    def get_impact_score(self, concept_id: str) -> float:
        dependents = 0
        for edge in self.edges:
            if edge.source_id == concept_id and edge.edge_type == "hierarchical":
                dependents += 1
        neighbor_count = len(self.adjacency.get(concept_id, set()))
        return dependents * 2.0 + neighbor_count * 0.5

    def get_edge_weight(self, a: str, b: str) -> float:
        return self._edge_weights.get((a, b), 0.0)

    def get_neighbors(self, concept_id: str) -> List[str]:
        return list(self.adjacency.get(concept_id, set()))

    def serialize(self) -> bytes:
        import json

        data = {
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "weight": e.weight,
                    "type": e.edge_type,
                }
                for e in self.edges
            ],
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> ConceptGraph:
        import json

        parsed = json.loads(data.decode("utf-8"))
        graph = cls()

        for e in parsed.get("edges", []):
            edge = ConceptEdge(
                source_id=e["source"],
                target_id=e["target"],
                weight=e["weight"],
                edge_type=e["type"],
            )
            graph.edges.append(edge)
            graph._edge_weights[(e["source"], e["target"])] = e["weight"]
            graph._edge_weights[(e["target"], e["source"])] = e["weight"]

        for k, neighbors in parsed.get("adjacency", {}).items():
            graph.adjacency[k] = set(neighbors)

        return graph