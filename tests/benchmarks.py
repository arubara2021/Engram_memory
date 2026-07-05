from __future__ import annotations

import os
import sys
import time
from statistics import mean, median
from typing import List

from engram.core.engine import Engram
from engram.core.config import EngramConfig
from engram.core.types import Chunk, ChunkResult
from engram.index.hash_table import (
    MultiHashTable,
    tokenize_for_concepts,
    hash_fnv1a,
    hash_murmur,
    hash_djb2,
)
from engram.index.builder import IndexBuilder
from engram.retrieve.hash_retriever import HashRetriever
from engram.retrieve.reranker import ReRanker


SAMPLE_TEXT = """
Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. These systems improve their performance over time without being
explicitly programmed. The field has grown rapidly in recent years due to advances in
computing power and the availability of large datasets.

The three main types of machine learning are supervised learning, unsupervised learning,
and reinforcement learning. Supervised learning uses labeled data to train models.
Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains
agents through trial and error with reward signals.

Chapter 2: Neural Networks

A neural network is a computing system inspired by biological neural networks. It consists
of layers of interconnected nodes called neurons. Each connection has a weight that adjusts
during training through backpropagation.

The input layer receives the raw data. Hidden layers process the information through
mathematical transformations using activation functions. The output layer produces the
final prediction or classification.

Deep learning refers to neural networks with multiple hidden layers. Convolutional neural
networks are specialized for image processing tasks. Recurrent neural networks handle
sequential data like text and time series. Transformers use self-attention mechanisms
for parallel processing of sequences.

Chapter 3: Training Process

Training a neural network involves forward propagation and backpropagation. During forward
propagation, input data flows through the network layer by layer to produce an output.
The loss function measures how far the output is from the expected result.

Backpropagation calculates gradients of the loss with respect to each weight in the
network. Gradient descent uses these gradients to update the weights iteratively. The
learning rate controls the size of each update step.

Overfitting occurs when a model learns the training data too well and fails to generalize
to new data. Regularization techniques like dropout, weight decay, and early stopping
help prevent overfitting. Cross-validation provides a reliable estimate of model
performance on unseen data.

Chapter 4: Evaluation Metrics

Classification accuracy measures the percentage of correct predictions. Precision measures
the proportion of true positives among predicted positives. Recall measures the proportion
of true positives among actual positives. The F1 score combines precision and recall
into a single metric.

For regression tasks, mean squared error measures average squared differences between
predictions and actual values. R-squared measures the proportion of variance explained
by the model. Mean absolute error provides a more robust measure less sensitive to
outliers.

Chapter 5: Feature Engineering

Feature engineering involves creating new input variables from raw data. Good features
can dramatically improve model performance. Common techniques include normalization,
standardization, one-hot encoding, and polynomial features.

Dimensionality reduction techniques like PCA and t-SNE help visualize high-dimensional
data and reduce computational requirements. Feature selection identifies the most
relevant features for a given task.
""".strip()


def _make_chunks_from_text(text: str) -> List[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        words = tokenize_for_concepts(para)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{i}",
                doc_id="bench",
                text=para,
                word_count=len(para.split()),
                page=0,
                section="",
                index=i,
                words=words,
                original_text=para,
            )
        )
    return chunks


def bench_hash_functions(iterations: int = 100000) -> None:
    test_strings = [
        "attention mechanism",
        "neural network",
        "machine learning algorithm",
        "backpropagation gradient descent",
        "convolutional neural network architecture",
    ]

    print("\n  Hash Function Benchmark")
    print(f"  {'Function':<12s} {'Total (ms)':>12s} {'Per hash (ns)':>14s} {'Hashes/sec':>12s}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 14} {'-' * 12}")

    for name, fn in [("FNV-1a", hash_fnv1a), ("Murmur3", hash_murmur), ("DJB2", hash_djb2)]:
        start = time.perf_counter()
        for _ in range(iterations):
            for s in test_strings:
                fn(s, 65536)
        elapsed = (time.perf_counter() - start) * 1000
        total_hashes = iterations * len(test_strings)
        per_hash_ns = elapsed / total_hashes * 1_000_000
        hashes_per_sec = total_hashes / (elapsed / 1000)
        print(f"  {name:<12s} {elapsed:>10.1f}ms {per_hash_ns:>12.1f}ns {hashes_per_sec:>10.0f}")


def bench_table_build() -> None:
    print("\n  Table Build Benchmark")
    chunks = _make_chunks_from_text(SAMPLE_TEXT)

    times = []
    for _ in range(5):
        table = MultiHashTable()
        start = time.perf_counter()
        table.build_from_chunks(chunks)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = mean(times)
    stats = table.get_stats()
    print(f"  Chunks:             {len(chunks)}")
    print(f"  Unique N-grams:     {stats.total_unique_ngrams:,}")
    print(f"  Total insertions:   {stats.total_entries:,}")
    print(f"  Table size:         {stats.table_size:,}")
    print(f"  Load factor:        {stats.load_factor:.3f}")
    print(f"  Avg build time:     {avg_time:.1f}ms")
    print(f"  Collision rate:     {mean(stats.collision_rates):.1f}%")


def bench_lookup_speed() -> None:
    print("\n  Lookup Speed Benchmark (Engram vs Brute Force)")
    chunks = _make_chunks_from_text(SAMPLE_TEXT)
    table = MultiHashTable()
    table.build_from_chunks(chunks)

    queries = [
        "machine learning",
        "neural network",
        "backpropagation",
        "gradient descent",
        "overfitting regularization",
        "convolutional neural network",
        "activation function",
        "loss function",
        "feature engineering",
        "precision recall",
    ]

    engram_times: List[float] = []
    brute_times: List[float] = []

    for query in queries:
        start = time.perf_counter()
        results = table.lookup_by_ngram(query)
        elapsed = (time.perf_counter() - start) * 1000
        engram_times.append(elapsed)

        start = time.perf_counter()
        query_tokens = set(tokenize_for_concepts(query))
        for chunk in chunks:
            overlap = len(query_tokens & set(chunk.words))
        elapsed = (time.perf_counter() - start) * 1000
        brute_times.append(elapsed)

    engram_avg = mean(engram_times)
    brute_avg = mean(brute_times)
    speedup = brute_avg / max(engram_avg, 0.001)

    print(f"  Queries:            {len(queries)}")
    print(f"  Chunks:             {len(chunks)}")
    print(f"  Engram avg:         {engram_avg:.3f}ms")
    print(f"  Engram median:      {median(engram_times):.3f}ms")
    print(f"  Brute force avg:    {brute_avg:.3f}ms")
    print(f"  Brute force median: {median(brute_times):.3f}ms")
    print(f"  Speedup:            {speedup:.1f}x")


def bench_concept_extraction() -> None:
    print("\n  Concept Extraction Benchmark")
    chunks = _make_chunks_from_text(SAMPLE_TEXT)
    config = EngramConfig(max_concepts=30)

    times = []
    for _ in range(3):
        builder = IndexBuilder(config)
        start = time.perf_counter()
        state = builder.build(chunks, SAMPLE_TEXT)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = mean(times)
    print(f"  Chunks:             {len(chunks)}")
    print(f"  Concepts found:     {len(state.concepts)}")
    print(f"  Avg extract time:   {avg_time:.1f}ms")
    print(f"  Top 5 concepts:")
    for i, c in enumerate(state.concepts[:5]):
        print(f"    {i + 1}. {c.label:<30s} score={c.score:.1f} freq={c.frequency}")


def bench_full_pipeline() -> None:
    print("\n  Full Pipeline Benchmark")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_TEXT)
        tmp_path = f.name

    try:
        config = EngramConfig(
            chunk_strategy="recursive",
            max_concepts=20,
            final_top_k=5,
        )
        engine = Engram(config)

        ingest_times = []
        retrieve_times = []

        for _ in range(3):
            engine.documents.clear()
            start = time.perf_counter()
            doc_id = engine.ingest(tmp_path)
            elapsed = (time.perf_counter() - start) * 1000
            ingest_times.append(elapsed)

            queries = [
                "What is machine learning?",
                "How do neural networks work?",
                "Explain backpropagation",
                "What causes overfitting?",
                "How to evaluate classification models?",
            ]

            for query in queries:
                start = time.perf_counter()
                results = engine.retrieve(query)
                elapsed = (time.perf_counter() - start) * 1000
                retrieve_times.append(elapsed)

        print(f"  Avg ingest time:    {mean(ingest_times):.1f}ms")
        print(f"  Avg retrieve time:  {mean(retrieve_times):.3f}ms")
        print(f"  Median retrieve:    {median(retrieve_times):.3f}ms")
        print(f"  Min retrieve:       {min(retrieve_times):.3f}ms")
        print(f"  Max retrieve:       {max(retrieve_times):.3f}ms")
        print(f"  Total queries:      {len(retrieve_times)}")

    finally:
        os.unlink(tmp_path)


def bench_serialization() -> None:
    print("\n  Serialization Benchmark")
    chunks = _make_chunks_from_text(SAMPLE_TEXT)
    config = EngramConfig(max_concepts=20)
    builder = IndexBuilder(config)
    state = builder.build(chunks, SAMPLE_TEXT)

    table = state.hash_tables

    serialize_times = []
    deserialize_times = []

    for _ in range(3):
        start = time.perf_counter()
        data = table.serialize()
        elapsed = (time.perf_counter() - start) * 1000
        serialize_times.append(elapsed)

        start = time.perf_counter()
        MultiHashTable.deserialize(data)
        elapsed = (time.perf_counter() - start) * 1000
        deserialize_times.append(elapsed)

    print(f"  Data size:          {len(data):,} bytes ({len(data) / 1024:.1f} KB)")
    print(f"  Avg serialize:      {mean(serialize_times):.1f}ms")
    print(f"  Avg deserialize:    {mean(deserialize_times):.1f}ms")


def run_all_benchmarks() -> None:
    print("=" * 60)
    print("  ENGRAM BENCHMARKS")
    print("=" * 60)

    bench_hash_functions()
    bench_table_build()
    bench_lookup_speed()
    bench_concept_extraction()
    bench_serialization()
    bench_full_pipeline()

    print()
    print("=" * 60)
    print("  BENCHMARKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()