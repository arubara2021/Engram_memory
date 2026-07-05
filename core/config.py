from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EngramConfig:
    chunk_strategy: str = "recursive"
    chunk_min_words: int = 80
    chunk_max_words: int = 150
    chunk_overlap_sentences: int = 2
    hierarchical_small: int = 200
    hierarchical_large: int = 800
    semantic_threshold: float = 0.5

    hash_functions: List[str] = field(
        default_factory=lambda: ["fnv1a", "murmur3", "djb2"]
    )
    table_size_multiplier: float = 4.0
    target_load_factor: float = 0.35
    ngram_range: Tuple[int, int] = (1, 4)
    max_concepts: int = 0
    concept_min_freq: int = 2

    embedding_enabled: bool = False
    embedding_backend: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    similarity_threshold: float = 0.6
    use_hnsw: bool = False

    retrieval_method: str = "hash"
    hash_weight: float = 0.6
    vector_weight: float = 0.4
    rerank_top_k: int = 20
    final_top_k: int = 5
    false_positive_penalty: float = 0.3

    context_gate_enabled: bool = True
    gate_min_chunks: int = 2
    gate_max_chunks: int = 7
    gate_relevance_threshold: float = 0.3

    prefetch_enabled: bool = True
    prefetch_cache_size: int = 10
    prefetch_predict_count: int = 3

    llm_backend: str = "none"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_base_url: Optional[str] = None

    storage_path: str = "./engram_data"
    cache_enabled: bool = True
    compress_on_save: bool = True

    strip_references: bool = True
    max_reference_pages: int = 15
    strip_urls: bool = True
    min_chunk_words: int = 20
    max_chunk_words: int = 180
    parallel_ingest: bool = True
    max_workers: int = 4

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.chunk_min_words < 1:
            errors.append("chunk_min_words must be >= 1")
        if self.chunk_max_words <= self.chunk_min_words:
            errors.append("chunk_max_words must be > chunk_min_words")
        if self.target_load_factor <= 0 or self.target_load_factor >= 1:
            errors.append("target_load_factor must be between 0 and 1")
        if self.final_top_k < 1:
            errors.append("final_top_k must be >= 1")
        if self.rerank_top_k < self.final_top_k:
            errors.append("rerank_top_k must be >= final_top_k")
        if self.embedding_enabled and self.embedding_dimension < 1:
            errors.append("embedding_dimension must be >= 1")
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            errors.append("llm_temperature must be between 0 and 2")
        if self.gate_min_chunks < 1:
            errors.append("gate_min_chunks must be >= 1")
        if self.gate_max_chunks < self.gate_min_chunks:
            errors.append("gate_max_chunks must be >= gate_min_chunks")
        if self.max_chunk_words <= self.min_chunk_words:
            errors.append("max_chunk_words must be > min_chunk_words")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngramConfig:
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)