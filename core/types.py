from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class RetrieverMethod(str, Enum):
    HASH = "hash"
    VECTOR = "vector"
    DUAL = "dual"


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    PAGE = "page"
    CODE = "code"


class EmbeddingBackendType(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    NONE = "none"


class LLMBackendType(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    NONE = "none"


class DocumentDomain(str, Enum):
    ACADEMIC = "academic"
    BUSINESS = "business"
    FICTION = "fiction"
    SELF_HELP = "self_help"
    TECHNICAL = "technical"
    CODE = "code"
    LEGAL = "legal"
    MEDICAL = "medical"
    SCIENCE = "science"
    HISTORY = "history"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    full_text: str
    pages: List[Tuple[int, str]]
    metadata: Dict[str, Any]
    page_count: int
    total_chars: int
    total_words: int
    text_quality: float
    page_spans: List[Tuple[int, int, int]] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    full_text: str
    pages: int
    total_chars: int
    total_words: int
    text_quality: float
    domain: str
    language: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    word_count: int
    page: int
    section: str
    index: int
    embedding: Optional[List[float]] = None
    concepts: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    original_text: str = ""


@dataclass
class ChunkResult:
    chunk: Chunk
    score: float
    source: str
    hash_hits: int = 0
    vector_sim: float = 0.0
    rerank_score: float = 0.0
    match_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Concept:
    concept_id: str
    label: str
    ngrams: List[str]
    frequency: int
    chunk_ids: List[str]
    score: float
    tags: List[str] = field(default_factory=list)
    primary_chunk_id: Optional[str] = None
    definition: str = ""
    in_definition: bool = False
    in_heading: bool = False
    in_bold: bool = False
    word_count: int = 1


@dataclass
class ConceptEdge:
    source_id: str
    target_id: str
    weight: float
    edge_type: str


@dataclass
class Response:
    answer: str
    sources: List[ChunkResult]
    confidence: float
    model_used: str
    prompt_tokens: int = 0
    response_tokens: int = 0


@dataclass
class IndexState:
    doc_id: str
    chunks: List[Chunk]
    concepts: List[Concept]
    hash_tables: Any = None
    vector_store: Any = None
    concept_graph: Any = None
    ngram_index: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class DocumentState:
    document: Document
    index: IndexState


@dataclass
class SlotEntry:
    chunk_id: str
    ngram_text: str
    ngram_size: int
    hash_table_index: int
    relevance_score: float = 1.0
    position: int = 0


@dataclass
class ConceptSlot:
    concept_id: str
    terms: List[str]
    chunk_ids: List[str]
    primary_chunk_id: Optional[str] = None
    definition: str = ""


@dataclass
class EngramStats:
    total_entries: int
    total_unique_ngrams: int
    table_size: int
    table_sizes: List[int]
    collision_rates: List[float]
    concept_count: int
    build_time_ms: float
    avg_lookup_time_ms: float
    memory_usage_bytes: int
    load_factor: float


@dataclass
class GateDecision:
    context_level: str
    chunk_count: int
    difficulty: str
    include_prerequisites: bool


@dataclass
class RetrievalResult:
    chunk_ids: List[str]
    entries: List[SlotEntry]
    source: str
    lookup_time_ms: float
    gate_decision: Optional[GateDecision] = None
    re_rank_scores: List[float] = field(default_factory=list)