from __future__ import annotations

from .hashing import fnv1a, murmur3, djb2, hash_all
from .text import tokenize, normalize, sentence_split, word_count, extract_ngrams
from .math import cosine_similarity, dot_product, normalize_vector, tfidf_score
from .noise import is_url_noise, is_reference_line, filter_concept_noise
from .patterns import (
    DEFINITION_PATTERNS,
    HEADING_PATTERNS,
    REFERENCE_PATTERNS,
    CITATION_PATTERNS,
)
from .domain import DOMAIN_KEYWORDS, get_domain_keywords
from .metrics import Metrics
from .common_words import COMMON_WORDS, NOISE_TERMS, TECHNICAL_SUFFIXES

__all__ = [
    "fnv1a", "murmur3", "djb2", "hash_all",
    "tokenize", "normalize", "sentence_split", "word_count", "extract_ngrams",
    "cosine_similarity", "dot_product", "normalize_vector", "tfidf_score",
    "is_url_noise", "is_reference_line", "filter_concept_noise",
    "DEFINITION_PATTERNS", "HEADING_PATTERNS", "REFERENCE_PATTERNS", "CITATION_PATTERNS",
    "DOMAIN_KEYWORDS", "get_domain_keywords",
    "Metrics",
    "COMMON_WORDS", "NOISE_TERMS", "TECHNICAL_SUFFIXES",
]