from __future__ import annotations

import math
import struct
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.types import Chunk, Concept, ConceptSlot, EngramStats, SlotEntry


def hash_fnv1a(text: str, table_size: int) -> int:
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % table_size


def hash_murmur(text: str, table_size: int) -> int:
    h = 0x9747B28C
    data = text.encode("utf-8")
    for i, b in enumerate(data):
        h ^= b
        h = (h * 0x5BD1E995) & 0xFFFFFFFF
        h ^= (h >> 15)
    h ^= len(data)
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= (h >> 13)
    return h % table_size


def hash_djb2(text: str, table_size: int) -> int:
    h = 5381
    for b in text.encode("utf-8"):
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return h % table_size


HASH_FUNCTIONS = [hash_fnv1a, hash_murmur, hash_djb2]


def hash_all(text: str, table_size: int) -> List[int]:
    return [fn(text, table_size) for fn in HASH_FUNCTIONS]


def calculate_table_size(expected_entries: int, target_load: float = 0.35) -> int:
    min_size = int(expected_entries / target_load)
    power = 1
    while power < min_size:
        power *= 2
    return max(power, 4096)


COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their", "which",
    "what", "who", "whom", "whose", "about", "up", "down", "also", "been",
    "got", "get", "getting", "let", "like", "made", "now", "see", "way",
    "well", "back", "even", "still", "take", "come", "go", "know", "first",
    "last", "long", "great", "little", "old", "right", "big", "high",
    "small", "large", "next", "early", "young", "important", "however",
    "another", "upon", "often", "among", "whether", "within", "without",
    "given", "since", "part", "case", "time", "example", "section",
    "chapter", "respectively", "denote", "denotes", "note", "noted",
    "thus", "hence", "therefore", "furthermore", "moreover", "additionally",
    "specifically", "particularly", "generally", "typically", "usually",
    "simply", "already", "nearly", "almost", "rather", "quite", "instead",
    "either", "neither", "less", "every", "describe", "describes",
    "described", "consider", "considered", "obtain", "obtained", "achieve",
    "achieved", "provide", "provided", "represent", "represented",
    "correspond", "corresponding", "previous", "following", "left",
    "final", "main", "total", "full", "single", "possible", "available",
    "standard", "common", "specific", "certain", "similar", "various",
    "several", "additional", "separate", "entire", "special", "recent",
    "current", "original", "basic", "simple", "complex", "multiple",
    "two", "one", "new", "use", "used", "using", "set", "number",
    "table", "figure", "result", "results", "different", "based", "shown",
    "show", "type", "work", "make", "per", "much", "many", "best",
    "better", "good", "state", "data", "task", "tasks", "model", "models",
    "output", "input", "high", "low", "method", "methods",
}

NOISE_TERMS = {
    "http", "https", "url", "doi", "org", "arxiv", "abs", "pdf",
    "www", "com", "net", "edu", "vol", "pp", "isbn", "issn",
    "fig", "figure", "table", "appendix", "et", "al", "etal",
    "university", "institute", "department", "conference", "proceedings",
    "journal", "press", "pages", "ieee", "acm", "neurips", "icml",
    "iclr", "aaai", "acl", "emnlp", "naacl", "workshop", "symposium",
    "volume", "issue", "page", "submitted", "accepted", "published",
    "preprint", "manuscript", "draft",
}

URL_PATTERN = __import__("re").compile(
    r"https?://|www\.|\.org/|\.com/|\.edu/|\.net/|doi\.org|arxiv\.org|"
    r"arXiv:|DOI:|ISBN|ISSN|pp\.\s*\d|vol\.\s*\d|"
    r"\d{4}\.\d{4,5}|arXiv\s+\d{4}\.\d{4,5}",
    __import__("re").IGNORECASE,
)


def normalize_text(text: str) -> str:
    import re

    text = text.lower().strip()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_concepts(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = normalized.split()
    return [t for t in tokens if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1]


def is_url_noise(text: str) -> bool:
    import re

    if URL_PATTERN.search(text):
        return True
    words = text.lower().split()
    if not words:
        return False
    noise_count = sum(1 for w in words if w in NOISE_TERMS)
    if noise_count / len(words) > 0.4:
        return True
    return False


def extract_ngrams_from_tokens(tokens: List[str], max_n: int = 4, filter_boundaries: bool = True) -> List[Tuple[str, int]]:
    ngrams: List[Tuple[str, int]] = []
    n = len(tokens)
    for size in range(2, min(max_n + 1, n + 1)):
        for i in range(n - size + 1):
            gram_tokens = tokens[i : i + size]
            if filter_boundaries:
                first = gram_tokens[0].lower()
                last = gram_tokens[-1].lower()
                if first in COMMON_WORDS or last in COMMON_WORDS:
                    continue
                if first in NOISE_TERMS or last in NOISE_TERMS:
                    continue
            gram = " ".join(gram_tokens)
            ngrams.append((gram, size))
    return ngrams


def extract_all_indexable_ngrams(chunk: Chunk, max_n: int = 4) -> List[Tuple[str, int]]:
    tokens = chunk.words
    ngrams = extract_ngrams_from_tokens(tokens, max_n, filter_boundaries=True)
    for word in set(tokens):
        if word not in COMMON_WORDS and word not in NOISE_TERMS and len(word) > 4:
            ngrams.append((word, 1))
    return ngrams


class MultiHashTable:

    def __init__(self) -> None:
        self.table_size: int = 4096
        self.hash_count: int = 3
        self.tables: List[Dict[int, List[SlotEntry]]] = [
            defaultdict(list) for _ in range(3)
        ]
        self.concept_table: Dict[str, ConceptSlot] = {}
        self.total_insertions: int = 0
        self.collision_counts: List[int] = [0, 0, 0]
        self.ngram_to_chunks: Dict[str, Set[str]] = defaultdict(set)

    def build_from_chunks(
        self, chunks: List[Chunk], concepts: Optional[List[Dict]] = None
    ) -> float:
        unique_ngrams: Set[str] = set()
        all_ngrams_per_chunk: List[List[Tuple[str, int]]] = []

        for chunk in chunks:
            ngrams = extract_all_indexable_ngrams(chunk)
            all_ngrams_per_chunk.append(ngrams)
            for ng, _ in ngrams:
                unique_ngrams.add(ng)

        self.table_size = calculate_table_size(len(unique_ngrams), 0.35)
        self.tables = [defaultdict(list) for _ in range(3)]
        self.collision_counts = [0, 0, 0]
        self.total_insertions = 0
        self.ngram_to_chunks = defaultdict(set)

        start = time.perf_counter()

        for chunk_idx, chunk in enumerate(chunks):
            ngrams = all_ngrams_per_chunk[chunk_idx]
            for ngram_text, ngram_size in ngrams:
                indices = hash_all(ngram_text, self.table_size)
                self.ngram_to_chunks[ngram_text].add(chunk.chunk_id)

                for table_idx, slot_index in enumerate(indices):
                    entry = SlotEntry(
                        chunk_id=chunk.chunk_id,
                        ngram_text=ngram_text,
                        ngram_size=ngram_size,
                        hash_table_index=slot_index,
                        relevance_score=float(ngram_size),
                        position=chunk.index,
                    )
                    existing = self.tables[table_idx].get(slot_index, [])
                    if existing and existing[0].ngram_text != ngram_text:
                        self.collision_counts[table_idx] += 1
                    self.tables[table_idx][slot_index].append(entry)
                    self.total_insertions += 1

        if concepts:
            for concept in concepts:
                terms = [concept["term"]]
                for w in concept["term"].split():
                    if w not in COMMON_WORDS and w not in NOISE_TERMS and len(w) > 1:
                        terms.append(w)
                self.register_concept(
                    concept_id=concept["term"],
                    terms=terms,
                    chunk_ids=concept.get("chunk_ids", []),
                    primary_chunk_id=concept.get("primary_chunk_id"),
                    definition=concept.get("definition", ""),
                )

        build_time = (time.perf_counter() - start) * 1000
        return build_time

    def register_concept(
        self,
        concept_id: str,
        terms: List[str],
        chunk_ids: List[str],
        primary_chunk_id: Optional[str] = None,
        definition: str = "",
    ) -> None:
        self.concept_table[concept_id] = ConceptSlot(
            concept_id=concept_id,
            terms=terms,
            chunk_ids=chunk_ids,
            primary_chunk_id=primary_chunk_id or (chunk_ids[0] if chunk_ids else None),
            definition=definition,
        )

    def insert(self, key: str, value: SlotEntry) -> None:
        indices = hash_all(key, self.table_size)
        for table_idx, slot_index in enumerate(indices):
            existing = self.tables[table_idx].get(slot_index, [])
            if existing and existing[0].ngram_text != key:
                self.collision_counts[table_idx] += 1
            self.tables[table_idx][slot_index].append(value)
            self.total_insertions += 1

    def lookup(self, concept_id: str) -> List[SlotEntry]:
        concept = self.concept_table.get(concept_id)
        if not concept:
            return []
        results: List[SlotEntry] = []
        for i, chunk_id in enumerate(concept.chunk_ids):
            results.append(
                SlotEntry(
                    chunk_id=chunk_id,
                    ngram_text=concept_id,
                    ngram_size=0,
                    hash_table_index=-1,
                    relevance_score=1.0,
                    position=i,
                )
            )
        return results

    def lookup_by_ngram(self, text: str) -> List[SlotEntry]:
        query_tokens = tokenize_for_concepts(text)
        query_ngrams: Set[str] = set()
        for ng, _ in extract_ngrams_from_tokens(query_tokens, 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS and len(t) > 1:
                query_ngrams.add(t)

        if not query_ngrams:
            normalized = normalize_text(text)
            if normalized:
                query_ngrams = {normalized}

        chunk_matches: Dict[str, Dict] = {}

        for ngram_text in query_ngrams:
            indices = hash_all(ngram_text, self.table_size)
            for table_idx, slot_index in enumerate(indices):
                entries = self.tables[table_idx].get(slot_index, [])
                for entry in entries:
                    if entry.ngram_text == ngram_text:
                        cid = entry.chunk_id
                        if cid not in chunk_matches:
                            chunk_matches[cid] = {
                                "entry": entry,
                                "match_count": 0,
                                "matched_ngrams": set(),
                                "max_ngram_size": 0,
                            }
                        if ngram_text not in chunk_matches[cid]["matched_ngrams"]:
                            chunk_matches[cid]["match_count"] += 1
                            chunk_matches[cid]["matched_ngrams"].add(ngram_text)
                            chunk_matches[cid]["max_ngram_size"] = max(
                                chunk_matches[cid]["max_ngram_size"],
                                entry.ngram_size,
                            )

        results: List[SlotEntry] = []
        for cid, data in chunk_matches.items():
            entry = data["entry"]
            relevance = data["match_count"] / max(len(query_ngrams), 1)
            size_bonus = data["max_ngram_size"] * 0.1
            results.append(
                SlotEntry(
                    chunk_id=entry.chunk_id,
                    ngram_text=entry.ngram_text,
                    ngram_size=data["max_ngram_size"],
                    hash_table_index=entry.hash_table_index,
                    relevance_score=relevance + size_bonus,
                    position=entry.position,
                )
            )

        results.sort(key=lambda e: (-e.relevance_score, e.position))
        return results

    def lookup_hybrid(self, concept_id: str, query_text: str = "") -> List[SlotEntry]:
        concept_results = self.lookup(concept_id)
        ngram_results = self.lookup_by_ngram(query_text or concept_id)

        merged: Dict[str, SlotEntry] = {}
        for entry in concept_results + ngram_results:
            if entry.chunk_id not in merged or entry.relevance_score > merged[entry.chunk_id].relevance_score:
                merged[entry.chunk_id] = entry

        return sorted(merged.values(), key=lambda e: (-e.relevance_score, e.position))

    def lookup_with_expansion(
        self, concept: Dict, chunks: List[Chunk]
    ) -> List[SlotEntry]:
        concept_term = concept.get("term", "")
        results = self.lookup_hybrid(concept_term)
        target = 3

        if len(results) < target:
            tokens = concept_term.split()
            for token in tokens:
                if token not in COMMON_WORDS and token not in NOISE_TERMS and len(token) > 2:
                    unigram_results = self.lookup_by_ngram(token)
                    seen = {r.chunk_id for r in results}
                    for r in unigram_results:
                        if r.chunk_id not in seen:
                            r.relevance_score *= 0.7
                            results.append(r)
                            seen.add(r.chunk_id)
                    if len(results) >= target:
                        break

        if len(results) < target and concept.get("definition", ""):
            key_phrases = tokenize_for_concepts(concept["definition"][:300])
            expanded_query = " ".join(key_phrases[:10])
            if expanded_query.strip():
                expanded_results = self.lookup_by_ngram(expanded_query)
                seen = {r.chunk_id for r in results}
                for r in expanded_results:
                    if r.chunk_id not in seen:
                        r.relevance_score *= 0.5
                        results.append(r)
                        seen.add(r.chunk_id)

        if len(results) < target:
            seen = {r.chunk_id for r in results}
            for related_term in list(self.concept_table.keys())[:30]:
                if related_term != concept_term:
                    related = self.concept_table[related_term]
                    shared = set(concept.get("chunk_ids", [])) & set(related.chunk_ids)
                    if shared:
                        for r in self.lookup(related_term):
                            if r.chunk_id not in seen:
                                r.relevance_score *= 0.3
                                results.append(r)
                                seen.add(r.chunk_id)
                        if len(results) >= target:
                            break

        results.sort(key=lambda e: (-e.relevance_score, e.position))
        return results

    def re_rank(
        self,
        results: List[SlotEntry],
        query: str,
        concept: Optional[Dict] = None,
        chunks: Optional[List[Chunk]] = None,
    ) -> List[Tuple[SlotEntry, float]]:
        import re as _re

        query_tokens = set(tokenize_for_concepts(query))
        query_ngrams: Set[str] = set()
        for ng, _ in extract_ngrams_from_tokens(list(query_tokens), 4):
            query_ngrams.add(ng)
        for t in query_tokens:
            if t not in COMMON_WORDS and t not in NOISE_TERMS:
                query_ngrams.add(t)

        if not query_ngrams:
            return [(r, r.relevance_score) for r in results]

        chunk_map: Dict[str, Chunk] = {}
        if chunks:
            chunk_map = {c.chunk_id: c for c in chunks}

        concept_chunk_ids = set(concept.get("chunk_ids", [])) if concept else set()
        total_chunks = len(chunks) if chunks else 1

        scored: List[Tuple[SlotEntry, float]] = []
        for entry in results:
            chunk = chunk_map.get(entry.chunk_id)
            if not chunk:
                scored.append((entry, entry.relevance_score * 0.5))
                continue

            chunk_ngrams: Set[str] = set()
            for ng, _ in extract_ngrams_from_tokens(chunk.words, 4):
                chunk_ngrams.add(ng)
            for t in chunk.words:
                if t not in NOISE_TERMS:
                    chunk_ngrams.add(t)

            matching = query_ngrams & chunk_ngrams
            coverage_score = len(matching) / max(len(query_ngrams), 1)
            density_score = len(matching) / max(len(chunk_ngrams), 1) if chunk_ngrams else 0.0
            alignment_score = 1.0 if entry.chunk_id in concept_chunk_ids else 0.0
            position_score = 1.0 / (1.0 + chunk.index / max(total_chunks, 1))

            final_score = (
                0.35 * coverage_score
                + 0.30 * density_score
                + 0.25 * alignment_score
                + 0.10 * position_score
            )

            scored.append((entry, final_score))

        scored.sort(key=lambda x: -x[1])
        return scored

    def get_stats(self, build_time_ms: float = 0, avg_lookup_ms: float = 0) -> EngramStats:
        table_sizes = [len(t) for t in self.tables]
        collision_rates: List[float] = []
        for i, count in enumerate(self.collision_counts):
            size = table_sizes[i]
            rate = (count / size * 100) if size > 0 else 0
            collision_rates.append(round(rate, 2))

        memory = self.total_insertions * 120 + len(self.concept_table) * 400

        unique: Set[str] = set()
        for table in self.tables:
            for entries in table.values():
                for e in entries:
                    unique.add(e.ngram_text)

        load_factor = len(unique) / max(self.table_size, 1)

        return EngramStats(
            total_entries=self.total_insertions,
            total_unique_ngrams=len(unique),
            table_size=self.table_size,
            table_sizes=table_sizes,
            collision_rates=collision_rates,
            concept_count=len(self.concept_table),
            build_time_ms=build_time_ms,
            avg_lookup_time_ms=avg_lookup_ms,
            memory_usage_bytes=memory,
            load_factor=round(load_factor, 3),
        )

    def serialize(self) -> bytes:
        parts: List[bytes] = []
        parts.append(struct.pack("<I", self.table_size))
        parts.append(struct.pack("<I", self.total_insertions))

        for table_idx, table in enumerate(self.tables):
            parts.append(struct.pack("<I", len(table)))
            for slot_idx, entries in table.items():
                parts.append(struct.pack("<I", slot_idx))
                parts.append(struct.pack("<I", len(entries)))
                for entry in entries:
                    ngram_bytes = entry.ngram_text.encode("utf-8")
                    parts.append(struct.pack("<H", len(ngram_bytes)))
                    parts.append(ngram_bytes)
                    cid_bytes = entry.chunk_id.encode("utf-8")
                    parts.append(struct.pack("<H", len(cid_bytes)))
                    parts.append(cid_bytes)
                    parts.append(struct.pack("<I", entry.ngram_size))
                    parts.append(struct.pack("<f", entry.relevance_score))
                    parts.append(struct.pack("<I", entry.position))

        parts.append(struct.pack("<I", len(self.concept_table)))
        for cid, concept in self.concept_table.items():
            cid_bytes = cid.encode("utf-8")
            parts.append(struct.pack("<H", len(cid_bytes)))
            parts.append(cid_bytes)
            parts.append(struct.pack("<I", len(concept.terms)))
            for t in concept.terms:
                tb = t.encode("utf-8")
                parts.append(struct.pack("<H", len(tb)))
                parts.append(tb)
            parts.append(struct.pack("<I", len(concept.chunk_ids)))
            for ci in concept.chunk_ids:
                cb = ci.encode("utf-8")
                parts.append(struct.pack("<H", len(cb)))
                parts.append(cb)
            ppc = (concept.primary_chunk_id or "").encode("utf-8")
            parts.append(struct.pack("<H", len(ppc)))
            parts.append(ppc)

        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> MultiHashTable:
        table = cls()
        offset = 0

        def read(fmt: str, size: int):
            nonlocal offset
            val = struct.unpack_from(fmt, data, offset)[0]
            offset += size
            return val

        def read_bytes():
            nonlocal offset
            length = read("<H", 2)
            val = data[offset : offset + length].decode("utf-8")
            offset += length
            return val

        table.table_size = read("<I", 4)
        table.total_insertions = read("<I", 4)
        table.tables = [defaultdict(list) for _ in range(3)]

        for table_idx in range(3):
            num_slots = read("<I", 4)
            for _ in range(num_slots):
                slot_idx = read("<I", 4)
                num_entries = read("<I", 4)
                entries: List[SlotEntry] = []
                for _ in range(num_entries):
                    ngram = read_bytes()
                    chunk_id = read_bytes()
                    ngram_size = read("<I", 4)
                    relevance = read("<f", 4)
                    position = read("<I", 4)
                    entries.append(
                        SlotEntry(
                            chunk_id=chunk_id,
                            ngram_text=ngram,
                            ngram_size=ngram_size,
                            hash_table_index=slot_idx,
                            relevance_score=relevance,
                            position=position,
                        )
                    )
                table.tables[table_idx][slot_idx] = entries

        num_concepts = read("<I", 4)
        for _ in range(num_concepts):
            cid = read_bytes()
            num_terms = read("<I", 4)
            terms = [read_bytes() for _ in range(num_terms)]
            num_cids = read("<I", 4)
            chunk_ids = [read_bytes() for _ in range(num_cids)]
            ppc = read_bytes()
            table.concept_table[cid] = ConceptSlot(
                concept_id=cid,
                terms=terms,
                chunk_ids=chunk_ids,
                primary_chunk_id=ppc if ppc else None,
            )

        return table