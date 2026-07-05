from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def dot_product(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


def vector_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize_vector(v: List[float]) -> List[float]:
    norm = vector_norm(v)
    if norm == 0:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


def cosine_similarity_normalized(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    for i in range(n):
        dot += a[i] * b[i]
    return dot


def euclidean_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    total = 0.0
    for i in range(n):
        diff = a[i] - b[i]
        total += diff * diff
    return math.sqrt(total)


def tfidf_score(
    term_freq: float,
    doc_freq: int,
    total_docs: int,
) -> float:
    if total_docs <= 0 or doc_freq <= 0:
        return 0.0
    idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1
    return term_freq * idf


def compute_tfidf(
    tokens: List[str],
    all_doc_tokens: List[List[str]],
) -> Dict[str, float]:
    n = len(all_doc_tokens)
    if n == 0:
        return {}

    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    df: Dict[str, int] = {}
    for doc_tokens in all_doc_tokens:
        for t in set(doc_tokens):
            df[t] = df.get(t, 0) + 1

    scores: Dict[str, float] = {}
    for term, freq in tf.items():
        doc_f = df.get(term, 0)
        scores[term] = tfidf_score(float(freq), doc_f, n)

    return scores


def moving_average(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    result: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def exponential_decay(
    initial: float,
    time_elapsed: float,
    half_life: float,
) -> float:
    if half_life <= 0:
        return 0.0
    decay_rate = math.log(2) / half_life
    return initial * math.exp(-decay_rate * time_elapsed)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def weighted_average(
    values: List[float],
    weights: List[float],
) -> float:
    if not values or not weights:
        return 0.0
    n = min(len(values), len(weights))
    total_weight = sum(weights[:n])
    if total_weight == 0:
        return 0.0
    total = sum(v * w for v, w in zip(values[:n], weights[:n]))
    return total / total_weight