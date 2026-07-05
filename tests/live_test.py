import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engram.core.engine import Engram
from engram.core.config import EngramConfig


def separator(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def test_pdf(engine, filepath):
    filename = os.path.basename(filepath)
    separator(f"INGESTING: {filename}")

    if not os.path.exists(filepath):
        print(f"  FILE NOT FOUND: {filepath}")
        return None

    start = time.perf_counter()
    doc_id = engine.ingest(filepath)
    elapsed = (time.perf_counter() - start) * 1000

    state = engine.documents[doc_id]
    doc = state.document
    idx = state.index

    print(f"  Doc ID:          {doc_id}")
    print(f"  Ingest time:     {elapsed:.0f}ms")
    print(f"  Total words:     {doc.total_words:,}")
    print(f"  Total chars:     {doc.total_chars:,}")
    print(f"  Text quality:    {doc.text_quality:.2f}")
    print(f"  Chunks created:  {len(idx.chunks)}")
    print(f"  Concepts found:  {len(idx.concepts)}")

    pages_found = {}
    for chunk in idx.chunks:
        p = chunk.page
        if p not in pages_found:
            pages_found[p] = 0
        pages_found[p] += 1
    assigned = sum(v for k, v in pages_found.items() if k > 0)
    pages_found = {}
    for chunk in idx.chunks:
        p = chunk.page
        if p not in pages_found:
            pages_found[p] = 0
        pages_found[p] += 1
    assigned = sum(v for k, v in pages_found.items() if k > 0)
    print(f"  Pages tracked:   {assigned}/{len(idx.chunks)} chunks assigned")
    print(f"  Hash table:      {idx.hash_tables.total_insertions:,} entries")

    if idx.concepts:
        print(f"  Top 5 concepts:")
        for i, c in enumerate(idx.concepts[:5]):
            print(f"    {i+1}. {c.label:<35s} score={c.score:.1f}  freq={c.frequency}")

    return doc_id


def test_queries(engine, doc_id, queries):
    filename = doc_id
    separator(f"QUERYING: {filename}")

    for query in queries:
        start = time.perf_counter()
        results = engine.retrieve(query, top_k=3)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Q: {query}")
        print(f"  Time: {elapsed:.2f}ms  |  Results: {len(results)}")
        for i, r in enumerate(results):
            preview = r.chunk.text[:100].replace("\n", " ")
            page_str = f"p{r.chunk.page}" if r.chunk.page > 0 else "p?"
            print(f"    [{i+1}] score={r.score:.3f} [{page_str}] | {preview}...")
        print()


def main():
    config = EngramConfig(
        chunk_strategy="recursive",
        min_chunk_words=30,
        max_chunk_words=200,
        max_concepts=0,
        retrieval_method="hash",
        final_top_k=3,
    )

    engine = Engram(config)

    pdf_files = [
        r"C:\Users\arun0\Videos\CV\Attention_all_is_you_need.pdf",
        r"C:\Users\arun0\Videos\CV\DeepSeek_V4.pdf",
        r"C:\Users\arun0\Videos\CV\Rich_Dad_Poor_Dad.pdf",
        r"C:\Users\arun0\Videos\CV\Think-And-Grow-Rich_2011-06.pdf",
    ]

    doc_ids = []
    for pdf in pdf_files:
        doc_id = test_pdf(engine, pdf)
        if doc_id:
            doc_ids.append((doc_id, os.path.basename(pdf)))

    separator("GLOBAL STATS")
    stats = engine.get_stats()
    print(f"  Total documents:  {stats['documents']}")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Total concepts:   {stats['total_concepts']}")
    print(f"  Total words:      {stats['total_words']:,}")

    queries_per_doc = {
        "Attention_all_is_you_need.pdf": [
            "What is the attention mechanism?",
            "How do transformers work?",
            "What is self-attention?",
            "Explain multi-head attention",
            "What are positional encodings?",
        ],
        "DeepSeek_V4.pdf": [
            "What is DeepSeek?",
            "How does the model architecture work?",
            "What training methods are used?",
            "What are the key benchmarks?",
        ],
        "Rich_Dad_Poor_Dad.pdf": [
            "What is the main lesson about money?",
            "What is the difference between assets and liabilities?",
            "How does the rich dad think about investing?",
            "What is financial literacy?",
        ],
        "Think-And-Grow-Rich_2011-06.pdf": [
            "What is the secret to success?",
            "How does desire lead to achievement?",
            "What is the power of faith?",
            "How does the mastermind principle work?",
        ],
    }

    for doc_id, filename in doc_ids:
        queries = queries_per_doc.get(filename, [
            "What is the main topic?",
            "What are the key concepts?",
            "What conclusions are drawn?",
        ])
        test_queries(engine, doc_id, queries)

    separator("CROSS-DOCUMENT QUERIES")
    cross_queries = [
        "What is artificial intelligence?",
        "How to become wealthy?",
        "What is the best learning strategy?",
    ]
    for query in cross_queries:
        start = time.perf_counter()
        results = engine.retrieve(query, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Q: {query}")
        print(f"  Time: {elapsed:.2f}ms  |  Results: {len(results)}")
        for i, r in enumerate(results):
            preview = r.chunk.text[:80].replace("\n", " ")
            page_str = f"p{r.chunk.page}" if r.chunk.page > 0 else "p?"
            print(f"    [{i+1}] score={r.score:.3f} doc={r.chunk.doc_id} [{page_str}] | {preview}...")
        print()

    separator("DONE")
    print(f"  All {len(doc_ids)} documents processed successfully.")
    print()


if __name__ == "__main__":
    main()
