# Engram

**A universal RAG engine that builds permanent, queryable knowledge from any document.**

Engram ingests documents in any format, extracts structured knowledge, and retrieves the most relevant context for any query — all running locally with zero dependencies on cloud services.

---

## Table of Contents

- [What Engram Does](#what-engram-does)
- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Module Reference](#module-reference)
  - [Core Layer](#core-layer)
  - [Ingestion Layer](#ingestion-layer)
  - [Chunking Layer](#chunking-layer)
  - [Indexing Layer](#indexing-layer)
  - [Embedding Layer](#embedding-layer)
  - [Retrieval Layer](#retrieval-layer)
  - [Generation Layer](#generation-layer)
  - [Storage Layer](#storage-layer)
  - [Utilities Layer](#utilities-layer)
- [The Engram Pipeline](#the-engram-pipeline)
- [Ingestion Flow](#ingestion-flow)
- [Chunking Strategy](#chunking-strategy)
- [Indexing and Knowledge Extraction](#indexing-and-knowledge-extraction)
- [Hash-Based Retrieval](#hash-based-retrieval)
- [Vector-Based Retrieval](#vector-based-retrieval)
- [Dual Retrieval](#dual-retrieval)
- [Re-Ranking](#re-ranking)
- [Context Gate](#context-gate)
- [Query Expansion](#query-expansion)
- [Predictive Prefetching](#predictive-prefetching)
- [LLM Generation](#llm-generation)
- [Response Evaluation](#response-evaluation)
- [Persistence and Storage](#persistence-and-storage)
- [Project Management](#project-management)
- [Configuration](#configuration)
- [Supported Formats](#supported-formats)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Design Principles](#design-principles)

---

## What Engram Does

Engram transforms raw documents into a structured knowledge base that can be queried instantly. Unlike simple keyword search or basic vector similarity, Engram combines multiple retrieval strategies into a single unified pipeline that understands both exact matches and semantic meaning.

When you give Engram a document, it:

1. **Detects the format** automatically — PDF, HTML, Markdown, JSON, CSV, YAML, code files, plain text, and more.
2. **Parses and cleans** the content, fixing ligatures, removing noise, normalizing whitespace, and stripping reference sections.
3. **Splits the text into chunks** using a recursive strategy that respects sentence boundaries and natural document structure.
4. **Extracts concepts** using TF-IDF analysis to identify the most important terms and topics in the document.
5. **Builds a hash table** using triple hashing (FNV-1a, Murmur3, DJB2) to enable O(1) keyword lookup.
6. **Builds a concept graph** that maps relationships between extracted concepts based on co-occurrence.
7. **Optionally generates embeddings** for semantic similarity search.
8. **Indexes everything** into a unified state that can be queried instantly.

When you query Engram, it:

1. **Looks up the query** in the hash table for exact keyword matches.
2. **Optionally performs vector search** for semantic similarity.
3. **Merges results** from both strategies with intelligent scoring.
4. **Re-ranks** results based on coverage, density, alignment, and diversity.
5. **Applies a context gate** that controls how many chunks to return based on relevance.
6. **Expands the query** if initial results are insufficient.
7. **Returns the most relevant chunks** with detailed scoring metadata.

All of this happens locally, in milliseconds, with no network calls required for the core retrieval pipeline.

---

## How It Works

### The Problem

When you read a textbook and someone asks you a question, you don't scan every word linearly. You have a mental index — you know which chapter covers which topic, you remember key terms, and you can quickly locate relevant passages. Engram replicates this process computationally.

### The Approach

Engram uses a **dual-retrieval architecture** that mirrors how human memory works:

- **Hash-based retrieval** is like your keyword memory. You hear "neural network" and immediately think of Chapter 2. This is fast (O(1)) and exact.
- **Vector-based retrieval** is like your semantic memory. You hear "how do artificial brains learn" and connect it to neural networks even though the exact words are different. This captures meaning.
- **Dual retrieval** combines both, boosting results that appear in both strategies (just like how you're most confident about passages you remember both by keyword and by meaning).

The result is a system that finds relevant content whether the user uses the exact terminology in the document or phrases the question differently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        ENGRAM                           │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  INGEST  │→ │  CHUNK   │→ │  INDEX   │→ │ STORE  │  │
│  │          │  │          │  │          │  │        │  │
│  │ PDF      │  │ Recursive│  │ Hash     │  │ Binary │  │
│  │ HTML     │  │ Hierarch.│  │ Concepts │  │ File   │  │
│  │ Markdown │  │ Page     │  │ Ngrams   │  │ Project│  │
│  │ JSON     │  │ Overlap  │  │ Graph    │  │ Cache  │  │
│  │ CSV      │  │ Splitter │  │ Vector   │  │        │  │
│  │ YAML     │  │ RefStrip │  │          │  │        │  │
│  │ Code     │  │          │  │          │  │        │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ RETRIEVE │→ │ GENERATE │→ │ EVALUATE │→ │ RETURN │  │
│  │          │  │          │  │          │  │        │  │
│  │ Hash     │  │ DeepSeek │  │ Grounding│  │ Chunks │  │
│  │ Vector   │  │ OpenAI   │  │ Coverage │  │ Answer │  │
│  │ Dual     │  │ Ollama   │  │ Hallucin.│  │ Score  │  │
│  │ Rerank   │  │ Anthropic│  │ Confidence│ │ Meta   │  │
│  │ Expand   │  │ Prompt   │  │          │  │        │  │
│  │ Gate     │  │          │  │          │  │        │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
```

The system is organized into nine layers, each with a single responsibility:

| Layer | Module | Purpose |
|-------|--------|---------|
| Core | `engram/core` | Configuration, types, engine orchestration |
| Ingestion | `engram/ingest` | Format detection, parsing, cleaning |
| Chunking | `engram/chunk` | Text splitting with multiple strategies |
| Indexing | `engram/index` | Hash tables, concepts, n-grams, graphs |
| Embedding | `engram/embed` | Vector generation and similarity search |
| Retrieval | `engram/retrieve` | Query processing, ranking, context control |
| Generation | `engram/generate` | LLM backends, prompt building, evaluation |
| Storage | `engram/store` | Serialization, file I/O, project management |
| Utilities | `engram/utils` | Hashing, text processing, math, metrics |

---

## Module Reference

### Core Layer

The core layer defines the configuration system, data types, and the main engine that orchestrates everything.

**`core/config.py` — EngramConfig**

Central configuration object that controls every aspect of the pipeline. All parameters have sensible defaults, so you only need to configure what you want to change.

Key configuration groups:

- **Chunking**: `chunk_strategy` (recursive, hierarchical, page), `min_chunk_words`, `max_chunk_words`, `overlap_sentences`
- **Indexing**: `max_concepts`, `hash_load_factor`, `table_size`
- **Retrieval**: `retrieval_method` (hash, vector, dual), `final_top_k`, `rerank_top_k`
- **Generation**: `llm_backend`, `llm_api_key`, `llm_model`
- **Embedding**: `embedding_enabled`, `embedding_model`
- **Context Gate**: `gate_min_chunks`, `gate_max_chunks`, `gate_relevance_threshold`
- **Prefetch**: `prefetch_enabled`, `prefetch_cache_size`

**`core/types.py` — Data Types**

Every data structure in the system is defined as a dataclass for type safety and clarity:

- `Document`: Represents an ingested document with metadata, full text, and word count.
- `Chunk`: A piece of a document with its text, word list, position, page number, section, and optional embedding.
- `Concept`: An extracted knowledge unit with label, TF-IDF score, frequency, chunk references, and tags.
- `SlotEntry`: A hash table entry linking an n-gram to a chunk with relevance scoring.
- `ConceptSlot`: A hash table slot containing entries and concept metadata.
- `ChunkResult`: A retrieval result with the chunk, combined score, source (hash/vector/dual), and detailed match metadata.
- `IndexState`: The complete indexed state of a document — chunks, concepts, hash tables, vector store, concept graph, and n-gram index.
- `DocumentState`: Combines the document metadata with its indexed state.
- `Response`: A complete generation response with answer text, source chunks, confidence score, and evaluation data.

**`core/engine.py` — Engram**

The main engine class that users interact with. It orchestrates the entire pipeline:

1. Creates and configures all subsystems (chunker, indexer, retriever, etc.)
2. Provides `ingest()` to process a file or directory
3. Provides `retrieve()` to query the knowledge base
4. Provides `generate()` to get LLM-powered answers
5. Manages document lifecycle (list, remove, save, load)

The engine maintains an in-memory map of all indexed documents and their states, enabling instant retrieval across multiple documents.

---

### Ingestion Layer

The ingestion layer handles everything from detecting what kind of file you've given it to producing clean, parseable text.

**`ingest/auto.py` — FormatDetector**

Determines the format of a file using multiple strategies:

1. **Extension matching**: Maps file extensions to formats (`.pdf` → PDF, `.md` → Markdown, `.py` → code).
2. **Magic bytes**: Checks the first few bytes of the file for format signatures (PDF starts with `%PDF`, ZIP-based formats have specific headers).
3. **Content sniffing**: Reads the first 8KB and looks for patterns — HTML tags, JSON braces, YAML colons, CSV commas.
4. **Content-type mapping**: Accepts MIME types and maps them to internal format names.

The detector supports over 40 file extensions and can handle ambiguous cases by falling back to content analysis.

**`ingest/pdf.py` — PDFParser**

Extracts text from PDF files using PyMuPDF (fitz). Handles:

- Multi-page documents with page number tracking
- Section detection from headings
- Metadata extraction (title, author, page count)
- Text quality assessment based on the ratio of printable characters
- Graceful handling of scanned PDFs (low text quality detection)

**`ingest/markup.py` — MarkupParser**

Parses HTML and Markdown files:

- **HTML**: Strips all `<script>` and `<style>` tags, removes HTML attributes, extracts text content while preserving structure. Uses HTMLParser for standards-compliant parsing.
- **Markdown**: Removes code block markers (while preserving the code content), strips link syntax (keeping the text), removes image tags, and extracts heading structure for metadata.
- **RST**: Basic reStructuredText support with directive and reference cleanup.

**`ingest/data.py` — DataParser**

Handles structured data formats:

- **JSON**: Recursively walks the structure, flattening keys and values into readable text. Arrays of objects get formatted as labeled entries.
- **JSONL**: Parses each line as a separate JSON object and combines them.
- **CSV**: Reads with the csv module, formats each row with column headers as labels. Detects encoding and handles different delimiters.
- **YAML**: Recursively walks the YAML structure similar to JSON handling.
- **TOML**: Converts TOML sections into labeled key-value text.

**`ingest/text.py` — TextParser**

Handles plain text and log files:

- Detects log format by looking for timestamp patterns (ISO dates, common log formats)
- Extracts log entry count and time ranges for metadata
- Falls back to basic text parsing for generic `.txt` files
- Supports `.eml` (email) files

**`ingest/code.py` — CodeParser**

Parses source code files in any programming language:

- Extracts the file structure: imports, classes, methods, functions
- Generates a structural overview that describes the code organization
- Extracts docstrings and comments as contextual information
- Identifies class hierarchies and method signatures
- Produces both the raw code and a structured description

**`ingest/cleaner.py` — TextCleaner**

Normalizes text after parsing:

1. Fixes ligatures (`ﬁ` → `fi`, `ﬂ` → `fl`)
2. Replaces smart quotes with standard quotes
3. Fixes hyphenation across line breaks (`informa- tion` → `information`)
4. Normalizes whitespace (tabs to spaces, collapses multiple spaces)
5. Removes control characters while preserving newlines
6. Strips zero-width characters

**`ingest/metadata.py` — MetadataDetector**

Extracts metadata from document text:

- **Title detection**: Looks for Markdown headings, first-line headings, ALL-CAPS titles, or falls back to filename.
- **Language detection**: Uses character frequency analysis and common word detection to identify the document language.
- **Domain detection**: Matches document vocabulary against domain keyword dictionaries (academic, technical, business, legal, medical, science, history, code, finance, self-help) and returns the best-matching domain.
- **Structure analysis**: Counts headings, paragraphs, lists, code blocks, and other structural elements.

**`ingest/base.py` — BaseParser**

Abstract base class that all parsers inherit from. Defines the interface:

- `parse(filepath) → ParseResult`: Returns a ParseResult with full text, cleaned text, metadata, quality score, and word/character counts.
- `supports(format) → bool`: Whether this parser handles the given format.

---

### Chunking Layer

The chunking layer splits parsed text into appropriately-sized pieces for indexing. Different strategies handle different document types.

**`chunk/recursive.py` — RecursiveChunker**

The default and most sophisticated chunking strategy. It works like how you'd naturally break up a document:

1. **Paragraph split**: First tries splitting on double newlines (paragraph boundaries).
2. **Sentence split**: If a paragraph is too large, splits it into sentences.
3. **Word split**: If a sentence is still too large, splits on word boundaries.
4. **Merge small**: If chunks are too small, merges them with adjacent chunks until they reach the minimum word count.

The chunker respects `min_chunk_words` and `max_chunk_words` from configuration. Each chunk gets a unique ID (`chunk_0`, `chunk_1`, etc.), the normalized word list, original text, and position metadata.

**`chunk/hierarchical.py` — HierarchicalChunker**

Creates two levels of chunks for multi-granularity retrieval:

- **Small chunks** (~80 words): Fine-grained pieces for precise matching.
- **Large chunks** (~200 words): Broader context windows that contain the small chunks.

Both small and large chunks are returned together. During retrieval, a small chunk match gives precise relevance, while large chunks provide surrounding context. This mirrors how you might remember a specific sentence but also want to re-read the whole paragraph.

**`chunk/page.py` — PageChunker**

Splits documents by page boundaries, useful for PDFs and documents with explicit page structure:

- Respects page numbers from the source document
- Merges pages that are too short (fewer than `min_chunk_words`)
- Splits pages that are too long using sentence boundaries
- Each chunk retains its page number for source attribution

**`chunk/overlap.py` — OverlapHandler**

Adds overlap between consecutive chunks to prevent context loss at chunk boundaries:

- Takes the last N sentences from each chunk and prepends them to the next chunk
- The first chunk remains unchanged (no preceding context to borrow)
- Overlap size is configurable via `overlap_sentences` (default: 2)
- Overlap helps retrieval find relevant content that spans chunk boundaries

**`chunk/splitter.py` — SentenceSplitter**

Splits text into sentences while protecting special patterns:

- Protects URLs from being split at periods (`https://example.com` stays intact)
- Protects decimal numbers (`3.14` stays intact)
- Protects email addresses and abbreviations
- Handles multiple punctuation marks
- Merges very short sentences (fewer than 8 words) with their neighbors

**`chunk/reference.py` — ReferenceStripper**

Removes bibliography and reference sections from documents:

- Detects reference section headers ("References", "Bibliography", "Works Cited")
- Identifies reference lines by author-date patterns
- Scans from the end of the document to find trailing reference sections
- Returns the document text without the references

**`chunk/factory.py` — ChunkerFactory**

Creates chunker instances by name. Supports:

- `"recursive"` → RecursiveChunker
- `"hierarchical"` → HierarchicalChunker
- `"page"` → PageChunker

The factory is extensible — you can register custom chunker classes.

---

### Indexing Layer

The indexing layer builds the data structures that enable fast retrieval. This is where raw text becomes queryable knowledge.

**`index/hash_table.py` — MultiHashTable**

The heart of Engram's O(1) retrieval. Uses **three independent hash functions** (FNV-1a, Murmur3, DJB2) to map n-grams to chunk references:

**How it works:**

1. Each chunk's words are extracted and normalized.
2. N-grams of sizes 2, 3, and 4 are generated from the word list.
3. Each n-gram is hashed by all three hash functions.
4. The hash values index into three separate tables.
5. Each table slot stores the n-gram text, chunk ID, n-gram size, and relevance score.

**Why three hash functions?**

A single hash function can have collisions (two different n-grams mapping to the same slot). With three independent hash functions, a true match appears in all three tables. A false positive would need to collide in all three, which is extremely unlikely. This triple-verification gives high confidence that a lookup result is genuine.

**Lookup process:**

When you search for "neural network":

1. The query is tokenized and normalized.
2. N-grams are extracted from the query tokens.
3. Each n-gram is looked up in all three tables.
4. Only chunks that appear for the same n-gram across tables are returned.
5. Results are scored by the number of matching n-grams relative to the total query n-grams.

The table size is calculated as a power of 2 based on the expected number of entries divided by the target load factor, ensuring good distribution.

**Concept registration:**

In addition to n-gram entries, the hash table maintains a concept registry. Concepts (extracted by the ConceptExtractor) are registered with their n-grams and primary chunk IDs. This allows direct concept lookup alongside n-gram lookup.

**`index/concept.py` — ConceptExtractor**

Extracts the most important concepts from a document using a multi-factor scoring system:

1. **TF-IDF base**: Terms that appear frequently in this document but rarely in general text get high scores.
2. **Bonus factors**:
   - Terms appearing in headings get a score bonus (headings indicate important topics).
   - Terms appearing in bold or italicized text get a bonus (authors emphasize key terms).
   - Multi-word terms get a length bonus (longer terms are usually more specific and meaningful).
   - Terms appearing in definition patterns get a bonus ("X is defined as...", "X refers to...").
3. **Noise filtering**: Common words, noise terms (URLs, references), and very short terms are filtered out.
4. **Term cleaning**: Strips leading articles, prepositions, and other noise from extracted terms.

Each concept gets:
- A unique ID derived from the normalized label
- A TF-IDF score weighted by all bonus factors
- A list of chunk IDs where it appears
- A frequency count across the document
- Tags indicating where it was found (D = definition, H = heading, B = bold)
- The ID of the primary (highest-scoring) chunk

**`index/ngram.py` — NgramIndexer**

Builds a reverse n-gram index that maps each unique n-gram to the set of chunk IDs where it appears. This index is stored in the `IndexState` and used for quick lookups and debugging.

**`index/graph.py` — ConceptGraph**

Builds a relationship graph between concepts based on co-occurrence:

1. For each pair of concepts, count how many chunks they share.
2. If two concepts co-occur in enough chunks, create an edge between them.
3. The edge weight is based on the number of shared chunks.
4. The graph supports traversal via `get_related(term, max_depth)` to find concepts related through chains of co-occurrence.

The concept graph enables:
- **Query expansion**: If a query matches concept A, and concept B is closely related to A, the system can also look for chunks containing B.
- **Prefetching**: Predicting which concepts the user will ask about next based on their query history and the graph.
- **Context enrichment**: Understanding the broader topic structure of the document.

**`index/builder.py` — IndexBuilder**

Orchestrates the entire indexing process:

1. Creates the hash table and builds it from all chunks.
2. Extracts concepts using TF-IDF analysis.
3. Registers concepts in the hash table.
4. Builds the n-gram index.
5. Builds the concept graph.
6. Returns the complete `IndexState`.

This is the single entry point for turning chunks into a queryable knowledge base.

**`index/cache.py` — ApproximateCache**

A cache that supports fuzzy matching on query keys:

- Stores query-result pairs with a similarity threshold.
- On lookup, compares the new query against all cached queries using word-overlap similarity.
- If a cached query is similar enough (above the threshold), returns the cached result.
- Uses LRU eviction when the cache is full.
- Useful for avoiding redundant retrieval when users ask very similar questions.

---

### Embedding Layer

The embedding layer converts text into dense vector representations for semantic similarity search.

**`embed/store.py` — VectorStore**

Stores document chunk embeddings and supports cosine similarity search:

- Stores vectors as lists of floats with an associated chunk ID and optional metadata.
- Search computes cosine similarity between the query vector and all stored vectors.
- Returns results sorted by similarity with optional threshold filtering.
- Supports both JSON serialization (for human-readable storage) and binary serialization (for compact storage using struct packing).
- Designed for moderate-scale use (thousands of vectors). For millions of vectors, an external ANN index would be needed.

**`embed/local.py` — LocalEmbedder**

Generates embeddings locally using sentence-transformers:

- Loads a model like `all-MiniLM-L6-v2` (384 dimensions) for fast, high-quality embeddings.
- Falls back to a TF-IDF-based embedding if sentence-transformers is not installed.
- The TF-IDF fallback uses scikit-learn's TfidfVectorizer to create sparse vectors, which are then converted to dense lists.
- Supports batch embedding for efficiency.

**`embed/remote.py` — RemoteEmbedder**

Generates embeddings via a remote API:

- Sends text to an HTTP endpoint and receives vector responses.
- Supports any API that accepts `{"input": "text"}` and returns `{"data": [{"embedding": [...]}]}`.
- Handles authentication via API key headers.
- Useful for accessing larger embedding models hosted on dedicated servers.

**`embed/pipeline.py` — EmbeddingPipeline**

Manages the embedding lifecycle:

- Holds a reference to the active embedding backend (local or remote).
- Provides `embed_query()` for single-query embedding.
- Provides `embed_chunks()` to embed a list of chunks and store their vectors.
- Provides `search()` to find the most similar chunks to a query.
- Tracks the embedding dimension and number of stored vectors.

---

### Retrieval Layer

The retrieval layer takes a user query and returns the most relevant chunks. This is where Engram's dual-retrieval architecture comes together.

**`retrieve/hash_retriever.py` — HashRetriever**

Performs O(1) keyword-based retrieval:

1. Tokenizes the query into words and n-grams (sizes 2, 3, 4).
2. Looks up each n-gram in the hash table using all three hash functions.
3. Counts how many query n-grams each chunk matches.
4. Calculates a score: `matched_ngrams / total_query_ngrams`.
5. Penalizes single-ngram matches for multi-word queries (reducing false positives).
6. Adds a bonus for larger n-gram matches (a 4-gram match is more valuable than a 2-gram match).

The retriever is fast because hash table lookups are O(1), regardless of how many chunks exist.

**`retrieve/vector_retriever.py` — VectorRetriever**

Performs semantic similarity retrieval:

1. Embeds the query using the embedding pipeline.
2. Searches the vector store for the most similar chunk embeddings.
3. Returns results with cosine similarity scores.

This is slower than hash retrieval (requires computing similarity against all stored vectors) but captures meaning rather than just keywords.

**`retrieve/dual.py` — DualRetriever**

Merges results from hash and vector retrieval:

- For each chunk, tracks whether it was found by hash, vector, or both.
- Chunks found by **both** strategies get a 20% score boost (high confidence).
- Chunks found only by hash use the hash weight (default 0.6).
- Chunks found only by vector use the vector weight (default 0.4).
- Results are combined and sorted by the final dual score.

The weighting reflects that exact keyword matches (hash) are typically more reliable than semantic similarity (vector) for factual queries, but combining both gives the best results.

**`retrieve/reranker.py` — ReRanker**

Applies multi-layer scoring to refine retrieval results:

**Scoring factors:**

1. **Coverage** (35% weight): What fraction of the query's n-grams appear in the chunk? Higher coverage means the chunk directly addresses more of the query.
2. **Density** (30% weight): How concentrated are the matching n-grams within the chunk? A chunk with many matching n-grams relative to its total n-grams is more focused and relevant.
3. **Alignment** (25% weight): Does the chunk belong to a concept that was directly mentioned in the query? Concept-aligned chunks get a bonus.
4. **Position** (10% weight): Earlier chunks in the document get a slight bonus, reflecting the tendency for documents to introduce key topics early.

**Diversity control:**

After scoring, the reranker applies diversity filtering to avoid returning multiple chunks that say essentially the same thing. If two chunks share more than 85% of their first 30 words, only the higher-scoring one is kept.

During scoring, chunks that overlap heavily with other results receive a diversity penalty (up to 0.2 for 80%+ overlap), preventing redundant content from dominating the results.

**`retrieve/context_gate.py` — ContextGate**

Controls how many chunks are returned based on relevance quality:

- **High relevance** (score > 0.7): Returns fewer chunks (min_chunks), because the top results are very targeted.
- **Medium relevance** (0.4-0.7): Returns a moderate number of chunks.
- **Low relevance** (0.2-0.4): Returns more chunks to give broader context.
- **Very low relevance** (< 0.2): Returns up to max_chunks, since nothing is a great match and broader coverage helps.

The gate always returns at least `min_chunks` (default 2) and at most `max_chunks` (default 7). This prevents both information starvation and context overload.

**`retrieve/expansion.py` — QueryExpander**

Falls back to expanded retrieval when initial results are insufficient:

**Expansion strategies (applied in order):**

1. **Unigram expansion**: Breaks the query into individual words and searches for each, finding chunks that match any word in the query.
2. **Concept term expansion**: Looks for concepts whose definitions share words with the query, then searches for chunks containing those related concepts.
3. **Graph expansion**: Uses the concept graph to find concepts related to query-matched concepts, then searches for chunks containing those graph-related concepts.

Each expansion strategy applies a decay factor to the score (0.7, 0.5, 0.3 respectively), so direct matches always rank above expanded matches. This ensures that expanded results supplement rather than replace precise matches.

**`retrieve/prefetch.py` — Prefetcher**

Predicts and pre-caches future queries:

**Prediction strategy:**

1. Tracks the user's query history.
2. When a query matches a concept, uses the concept graph to predict related concepts the user might ask about next.
3. Pre-fetches retrieval results for predicted concepts and caches them.
4. When the predicted query actually arrives, it's served from cache instantly.

The prefetcher uses an LRU cache, so frequently accessed queries stay cached while stale predictions are evicted. The cache hit rate is tracked to measure prediction accuracy.

**`retrieve/pipeline.py` — RetrievalPipeline**

Orchestrates the entire retrieval process:

1. Checks the prefetch cache for instant results.
2. Runs hash retrieval and/or vector retrieval based on configuration.
3. Merges results via the DualRetriever if using dual mode.
4. Falls back to query expansion if too few results.
5. Re-ranks all results.
6. Applies the context gate.
7. Deduplicates results.
8. Caches the final results for future queries.
9. Triggers background prefetching for predicted queries.

---

### Generation Layer

The generation layer connects retrieval results to LLMs for natural language answers.

**`generate/base.py` — LLMBackend**

Abstract interface that all LLM backends implement:

- `generate(prompt, temperature, max_tokens) → str`: Generate a response.
- `is_available() → bool`: Check if the backend is configured and reachable.
- `generate_with_metadata(prompt, ...) → dict`: Generate with token counts and model info.
- `_truncate_prompt(prompt, max_chars)`: Prevent prompts from exceeding model limits.

**`generate/deepseek.py` — DeepSeekBackend**

Connects to the DeepSeek API:

- Uses the OpenAI-compatible chat completions endpoint.
- Reads the API key from config or the `DEEPSEEK_API_KEY` environment variable.
- Supports custom base URLs for proxied or alternative endpoints.
- Returns full metadata including token usage.

**`generate/openai.py` — OpenAIBackend**

Connects to the OpenAI API:

- Uses the standard `/v1/chat/completions` endpoint.
- Supports all GPT models (gpt-4o, gpt-4o-mini, etc.).
- Reads the API key from config or the `OPENAI_API_KEY` environment variable.
- Returns token usage and finish reason.

**`generate/ollama.py` — OllamaBackend**

Connects to a local Ollama instance:

- Uses both the `/api/generate` endpoint (for basic generation) and `/api/chat` endpoint (for system-prompt-aware generation).
- Auto-detects the model list from the local Ollama instance.
- Supports `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables.
- Defaults to `http://localhost:11434` with the `llama3` model.
- Returns timing information (total duration, eval duration) from the Ollama runtime.

**`generate/anthropic.py` — AnthropicBackend**

Connects to the Anthropic API:

- Uses the `/v1/messages` endpoint with the Anthropic-specific header format.
- Reads the API key from config or the `ANTHROPIC_API_KEY` environment variable.
- Handles the Anthropic content block format (extracting text from content arrays).
- Returns input/output token counts and stop reason.

**`generate/prompt.py` — PromptBuilder**

Constructs source-grounded prompts that prevent hallucination:

**RAG prompt structure:**

```
[System instructions: answer ONLY from source text]
SOURCE TEXT:
[1] (relevance: 0.92) The chunk text...
[2] (relevance: 0.85) Another chunk...
---
QUESTION: What is machine learning?
ANSWER:
```

The system instructions explicitly tell the LLM to:
- Answer only based on the provided source text.
- State "This information is not in the provided documents" when the answer isn't found.
- Cite which source chunk was used.

**Additional prompt types:**

- **Evaluation prompt**: Asks the LLM to compare a student's answer against source text.
- **Question generation prompt**: Generates study questions at recall, understanding, or application difficulty levels.
- **Summarization prompt**: Summarizes the source text while staying within the information provided.
- **No-context prompt**: Used when no relevant chunks were found, instructs the LLM to acknowledge the gap.

**`generate/evaluator.py` — ResponseEvaluator**

Checks whether an LLM response is grounded in the source:

**Evaluation metrics:**

1. **Grounding score**: Fraction of response sentences that share significant vocabulary with the source chunks. A sentence is considered grounded if 30%+ of its words appear in the source.
2. **Source coverage**: What fraction of source content is addressed in the response.
3. **Hallucination detection**: Sentences with 5+ words that aren't grounded are flagged as potential hallucinations.
4. **Uncertainty language detection**: Identifies when the model appropriately acknowledges knowledge limits ("This information is not in the provided documents").

**Confidence computation:**

Combines the average retrieval score with the chunk coverage ratio to produce an overall confidence score from 0 to 1.

**`generate/factory.py` — LLMFactory**

Creates LLM backend instances by name:

- Maps backend names to classes: `deepseek`, `openai`, `ollama`, `anthropic`.
- Supports registering custom backends.
- `get_available()` checks which backends are currently configured and reachable.
- `create_best_available()` tries backends in priority order and returns the first one that works.

---

### Storage Layer

The storage layer handles persistence — saving indexed documents to disk and loading them back.

**`store/serializer.py` — Serializer**

Converts the in-memory `IndexState` to and from binary format:

**Serialization process:**

1. Writes the document ID as a length-prefixed UTF-8 string.
2. Writes each chunk with all its fields (text, words, embeddings, metadata).
3. Writes each concept with all its fields (label, scores, chunk IDs, tags).
4. Writes the hash tables, vector store, and concept graph (if present).
5. Writes the n-gram index as JSON (for human readability in an otherwise binary format).
6. Compresses the entire binary blob with gzip.

**Why binary?**

Binary serialization is significantly faster and more compact than JSON for large data structures. A 10,000-chunk document that takes 50MB as JSON might be 5MB as binary and deserialize 10x faster.

**`store/filesystem.py` — FileStore**

Manages document storage on the local filesystem:

**Storage structure:**

```
engram_data/
├── doc_abc123/
│   ├── engram.bin          # Serialized index state
│   └── meta.json           # Human-readable metadata
├── doc_def456/
│   ├── engram.bin
│   └── meta.json
└── _projects/
    └── proj_xyz/
        └── project.json    # Project metadata
```

**Operations:**

- `save(doc_id, state, metadata)`: Serializes the index and writes both the binary file and metadata JSON.
- `load(doc_id)`: Reads the binary file and deserializes back to `IndexState`.
- `delete(doc_id)`: Removes the entire document directory.
- `list_documents()`: Scans the storage directory for valid indexed documents.
- `get_storage_size()`: Reports total disk usage per document and overall.
- `backup(path)` / `restore(path)`: Full backup and restore of the storage directory.

**`store/project.py` — ProjectStore**

Manages multi-document projects:

- A project is a named collection of documents with shared metadata.
- Documents are added to projects by reference (the actual index data stays in the document's storage directory).
- Projects track creation time, last update time, and per-document metadata.
- Supports listing all projects, viewing project statistics (total chunks, concepts, size), and deleting projects (optionally with their documents).

**`store/lru.py` — LRUCache**

A general-purpose LRU (Least Recently Used) cache:

- Standard put/get operations with O(1) time complexity using an OrderedDict.
- Tracks hit/miss counts and hit rate for monitoring.
- Supports resize, peek (without promoting), and bulk update.
- `get_or_compute()` provides memoization — computes the value only on cache miss.

---

### Utilities Layer

Shared utilities used throughout the codebase.

**`utils/hashing.py`**

Three hash functions optimized for n-gram distribution:

- **FNV-1a**: Fast, good distribution, uses XOR-fold multiply. Industry standard for hash tables.
- **Murmur3**: Excellent avalanche properties — small input changes cause large hash changes. Reduces clustering.
- **DJB2**: Simple and effective, uses shift-add multiply. Good secondary hash.

All three are pure Python with no external dependencies. The `hash_all()` convenience function returns all three hashes for a given input.

**`utils/text.py`**

Text processing utilities:

- `normalize()`: Lowercases, removes punctuation, fixes camelCase, replaces hyphens/underscores with spaces.
- `tokenize()`: Normalizes then splits on whitespace, filtering single-character tokens.
- `sentence_split()`: Splits text into sentences while protecting URLs, decimals, and abbreviations.
- `extract_ngrams()`: Generates n-grams of a given size from a token list.
- `fix_ligatures()`: Converts typographic ligatures to standard characters.
- `fix_hyphenation()`: Rejoins words split across lines with hyphens.

**`utils/math.py`**

Mathematical operations:

- `cosine_similarity()`: Computes cosine similarity between two vectors without external dependencies.
- `dot_product()`, `vector_norm()`, `normalize_vector()`: Basic vector operations.
- `tfidf_score()`: Computes TF-IDF weighting for a term.
- `sigmoid()`, `clamp()`, `weighted_average()`: Utility math functions.
- `exponential_decay()`: Used for time-based relevance decay.
- `moving_average()`: Smooths noisy data series.

**`utils/noise.py`**

Noise detection and filtering:

- `is_url_noise()`: Identifies text that's primarily URLs, DOIs, or reference metadata.
- `is_reference_line()`: Detects bibliography entries using author-date patterns.
- `filter_concept_noise()`: Validates concept terms, rejecting URLs, single characters, and pure-number terms.
- `filter_reference_section()`: Removes entire reference/bibliography sections from document text.
- `filter_url_lines()`: Removes lines that are primarily URL content.

**`utils/patterns.py`**

Pre-compiled regular expressions for fast pattern matching:

- `DEFINITION_PATTERNS`: 9 patterns that identify definitional text ("X is defined as...", "X refers to...", "X consists of...").
- `HEADING_PATTERNS`: Patterns for numbered headings, section labels, ALL-CAPS headings, and Markdown headings.
- `REFERENCE_PATTERNS`: Patterns for bibliography sections and reference entries.
- `CITATION_PATTERNS`: Patterns for in-text citations ([1], (Smith, 2020), DOI references).
- `TITLE_PATTERNS`: Patterns for document titles.

**`utils/domain.py`**

Domain keyword dictionaries for document classification:

Ten domains with 20-30 keywords each:

- **Academic**: abstract, methodology, hypothesis, literature, citation...
- **Technical**: algorithm, framework, architecture, deployment, pipeline...
- **Business**: revenue, strategy, stakeholders, quarterly, compliance...
- **Legal**: jurisdiction, liability, arbitration, testimony, statute...
- **Medical**: diagnosis, clinical, pharmaceutical, pathology, prognosis...
- **Science**: hypothesis, experiment, molecule, genome, quantum...
- **Self-help**: goal, habit, mindset, motivation, meditation...
- **History**: dynasty, empire, revolution, civilization, archaeology...
- **Code**: function, class, module, async, inheritance, polymorphism...
- **Finance**: stock, bond, equity, inflation, yield, margin...

`detect_domain()` scores the document against all domains and returns the best match.

**`utils/metrics.py`**

Performance tracking:

- **Timings**: Record operation durations with statistical analysis (avg, median, min, max, p95, p99).
- **Counters**: Track event counts (queries processed, cache hits, etc.).
- **Gauges**: Track current values (memory usage, queue depth, etc.).
- Supports context-manager timing (`with metrics.time_function("name"): ...`).
- `report()` produces a structured dictionary of all metrics.
- `summary()` produces a human-readable text summary.

**`utils/common_words.py`**

Word lists used for filtering:

- `COMMON_WORDS` (300+): Stop words and very common English words that don't carry retrieval value.
- `NOISE_TERMS` (50+): Terms that appear in document metadata but aren't content (http, doi, arxiv, university, copyright...).
- `TECHNICAL_SUFFIXES` (40+): Word endings that indicate technical terminology (-tion, -ment, -ness, -ology...).
- `DOMAIN_STOPWORDS` (40+): Words common in document contexts but not meaningful for retrieval (according, based, using, proposed, paper...).

---

## The Engram Pipeline

When you call `engine.ingest(filepath)` and then `engine.retrieve("question")`, here's exactly what happens at each step:

### Ingestion Flow

```
File on disk
    │
    ▼
FormatDetector.detect(filepath)
    │ → Returns: "pdf", "html", "json", "code", etc.
    ▼
Parser.parse(filepath)
    │ → Extracts raw text, metadata, quality score
    ▼
TextCleaner.clean(raw_text)
    │ → Fixes ligatures, smart quotes, hyphenation, whitespace
    ▼
MetadataDetector.detect(cleaned_text)
    │ → Extracts title, language, domain, structure
    ▼
ReferenceStripper.strip(cleaned_text)
    │ → Removes bibliography and reference sections
    ▼
Document(full_text, cleaned_text, metadata, words, ...)
```

### Chunking Flow

```
Cleaned text
    │
    ▼
ChunkerFactory.create(strategy)
    │
    ▼
RecursiveChunker.chunk(text, doc_id)
    │
    ├── Split on paragraphs (double newlines)
    │   ├── If paragraph > max_chunk_words:
    │   │   └── Split on sentences
    │   │       ├── If sentence > max_chunk_words:
    │   │       │   └── Split on words
    │   │       └── Merge adjacent short sentences
    │   └── If paragraph < min_chunk_words:
    │       └── Merge with next paragraph
    │
    ▼
OverlapHandler.apply(chunks, overlap_sentences)
    │ → Prepends last N sentences of chunk[i-1] to chunk[i]
    ▼
List[Chunk]
    │  Each chunk has: chunk_id, text, words, word_count, page, section, index
```

### Indexing Flow

```
List[Chunk]
    │
    ▼
MultiHashTable.build_from_chunks(chunks)
    │
    ├── For each chunk:
    │   ├── Extract n-grams (size 2, 3, 4) from chunk.words
    │   ├── For each n-gram:
    │   │   ├── Compute hash = FNV-1a(text, table_size)
    │   │   ├── Compute hash = Murmur3(text, table_size)
    │   │   ├── Compute hash = DJB2(text, table_size)
    │   │   └── Store SlotEntry in all three tables
    │   └── Track collision rates per table
    │
    ▼
ConceptExtractor.extract(chunks, full_text)
    │
    ├── Compute TF-IDF scores for all terms
    ├── Apply bonuses for headings, bold, definitions, multi-word
    ├── Filter noise and common words
    ├── Score and rank all candidate concepts
    └── Return top N concepts
    │
    ▼
MultiHashTable.register_concepts(concepts)
    │ → Register concept n-grams in the hash table
    │
    ▼
NgramIndexer.build_index(chunks, concepts, hash_tables)
    │ → Build reverse n-gram → chunk_ids mapping
    │
    ▼
ConceptGraph.build(concepts, chunks)
    │
    ├── For each pair of concepts:
    │   ├── Count shared chunks
    │   └── If overlap > threshold: create edge
    │
    ▼
IndexState(doc_id, chunks, concepts, hash_tables, ngram_index, concept_graph)
```

### Retrieval Flow

```
User query: "How do neural networks learn?"
    │
    ▼
Prefetcher.get(query)
    │ → Check cache for instant results
    │ → Cache miss: proceed with retrieval
    │
    ▼
[Hash Path]
HashRetriever.retrieve(query, hash_tables, chunks)
    │
    ├── Tokenize query → ["how", "neural", "networks", "learn"]
    ├── Extract n-grams → ["neural networks", "networks learn", "neural", "networks", "learn"]
    ├── For each n-gram:
    │   ├── hash_all(ngram, table_size) → [h1, h2, h3]
    │   ├── Look up in all three tables
    │   └── Collect matching chunk IDs
    ├── Score each chunk: matched_ngrams / total_query_ngrams
    └── Return top K results
    │
    ▼
[Vector Path] (if enabled)
VectorRetriever.retrieve(query, vector_store, chunks, embedding_pipeline)
    │
    ├── Embed query → query_vector
    ├── vector_store.search(query_vector, top_k, threshold)
    └── Return cosine-similarity-ranked results
    │
    ▼
DualRetriever.merge(hash_results + vector_results)
    │
    ├── For each unique chunk:
    │   ├── If found by BOTH: score = (0.6 * hash + 0.4 * vector) * 1.2
    │   ├── If found by hash only: score = hash * 0.6
    │   └── If found by vector only: score = vector * 0.4
    └── Sort by final score
    │
    ▼
[If too few results]
QueryExpander.expand(query, results, hash_tables, chunks, concepts, concept_graph)
    │
    ├── Unigram expansion (score * 0.7)
    ├── Concept term expansion (score * 0.5)
    └── Graph expansion (score * 0.3)
    │
    ▼
ReRanker.rerank(results, query, concept, chunks)
    │
    ├── For each result:
    │   ├── Coverage: matching_query_ngrams / total_query_ngrams
    │   ├── Density: matching_ngrams / total_chunk_ngrams
    │   ├── Alignment: 1.0 if chunk is in a query-matched concept
    │   ├── Position: 1 / (1 + chunk_index / total_chunks)
    │   ├── Diversity penalty: -0.1 to -0.2 for overlapping results
    │   └── Final = 0.35*coverage + 0.30*density + 0.25*alignment + 0.10*position - penalty
    │
    ├── Apply diversity filter (remove 85%+ overlapping chunks)
    └── Sort by rerank_score
    │
    ▼
ContextGate.apply(results)
    │
    ├── Filter by minimum relevance threshold (0.3)
    ├── Ensure at least min_chunks (2)
    ├── Ensure at most max_chunks (7)
    └── Return gated results
    │
    ▼
Prefetcher.put(query, results)
    │ → Cache results for future identical/similar queries
    │
    ▼
Background: Prefetcher.predict_from_history(concepts, concept_graph)
    │ → Predict and pre-cache next likely queries
    │
    ▼
List[ChunkResult]
    │  Each result has: chunk, score, source, hash_hits, vector_sim, rerank_score, match_details
```

### Generation Flow (Optional)

```
List[ChunkResult] + User question
    │
    ▼
PromptBuilder.build_rag_prompt(question, chunks)
    │
    │ → "You are a helpful assistant that answers based ONLY on source text..."
    │ → "SOURCE TEXT:"
    │ → "[1] (relevance: 0.92) Machine learning is a subset of AI..."
    │ → "[2] (relevance: 0.85) Neural networks consist of layers..."
    │ → "QUESTION: How do neural networks learn?"
    │ → "ANSWER:"
    │
    ▼
LLMBackend.generate(prompt, temperature, max_tokens)
    │
    ├── DeepSeekBackend → HTTP POST to DeepSeek API
    ├── OpenAIBackend → HTTP POST to OpenAI API
    ├── OllamaBackend → HTTP POST to local Ollama
    └── AnthropicBackend → HTTP POST to Anthropic API
    │
    ▼
ResponseEvaluator.evaluate(response, chunks)
    │
    ├── Grounding score: grounded_sentences / total_sentences
    ├── Source coverage: addressed_source_sentences / total_source_sentences
    ├── Hallucination flags: ungrounded sentences with 5+ words
    ├── Uncertainty detection: "not in the provided documents" patterns
    └── Confidence score: avg_retrieval_score * chunk_coverage
    │
    ▼
Response(text, sources, confidence, evaluation)
```

---

## Supported Formats

| Format | Extensions | Parser | Notes |
|--------|-----------|--------|-------|
| PDF | `.pdf` | PDFParser | Requires PyMuPDF (`fitz`) |
| HTML | `.html`, `.htm` | MarkupParser | Strips scripts, styles, tags |
| Markdown | `.md`, `.markdown` | MarkupParser | Preserves code blocks, removes markup |
| RST | `.rst` | MarkupParser | Basic reStructuredText support |
| JSON | `.json` | DataParser | Recursive structure flattening |
| JSONL | `.jsonl` | DataParser | Line-by-line JSON objects |
| CSV | `.csv` | DataParser | Header-labeled row formatting |
| YAML | `.yaml`, `.yml` | DataParser | Recursive structure flattening |
| TOML | `.toml` | DataParser | Section-labeled key-value format |
| Plain text | `.txt`, `.log`, `.eml` | TextParser | Log format auto-detection |
| Python | `.py` | CodeParser | Class/method extraction |
| JavaScript | `.js`, `.ts`, `.tsx`, `.jsx` | CodeParser | Function/class extraction |
| Java | `.java` | CodeParser | Class/method extraction |
| C/C++ | `.c`, `.cpp`, `.h`, `.hpp` | CodeParser | Function extraction |
| Rust | `.rs` | CodeParser | Struct/impl extraction |
| Go | `.go` | CodeParser | Struct/func extraction |
| Ruby | `.rb` | CodeParser | Class/method extraction |
| PHP | `.php` | CodeParser | Class/function extraction |
| Shell | `.sh`, `.bash` | CodeParser | Function extraction |
| SQL | `.sql` | CodeParser | Statement extraction |
| XML | `.xml` | MarkupParser | Tag stripping |
| LaTeX | `.tex` | TextParser | Command stripping |

---

## Quick Start

### Installation

```bash
pip install engram
```

### Basic Usage

```python
from engram import Engram

# Create the engine
engine = Engram()

# Ingest a document
doc_id = engine.ingest("my_notes.pdf")

# Query the knowledge base
results = engine.retrieve("What is the main argument?")

for r in results:
    print(f"Score: {r.score:.2f}")
    print(f"Text: {r.chunk.text[:200]}...")
    print()
```

### Ingest Multiple Documents

```python
engine = Engram()

# Ingest individual files
engine.ingest("chapter1.pdf")
engine.ingest("chapter2.pdf")
engine.ingest("notes.md")

# Or ingest an entire directory
engine.ingest("./my_documents/")

# Query across all documents
results = engine.retrieve("What are the key findings?")
```

### Generate Answers with LLM

```python
from engram import Engram, EngramConfig

config = EngramConfig(
    llm_backend="ollama",
    llm_model="llama3",
    final_top_k=5,
)

engine = Engram(config)
engine.ingest("research_paper.pdf")

# Get a complete response with sources
response = engine.generate("Summarize the methodology")
print(response.text)
print(f"Confidence: {response.confidence:.2f}")
print(f"Sources: {len(response.sources)} chunks")
```

### Custom Configuration

```python
from engram import Engram, EngramConfig

config = EngramConfig(
    # Chunking
    chunk_strategy="recursive",
    min_chunk_words=30,
    max_chunk_words=200,
    overlap_sentences=2,

    # Indexing
    max_concepts=25,

    # Retrieval
    retrieval_method="dual",
    final_top_k=5,
    rerank_top_k=20,

    # Context gate
    gate_min_chunks=2,
    gate_max_chunks=7,
    gate_relevance_threshold=0.3,

    # Generation
    llm_backend="deepseek",
    llm_model="deepseek-chat",
    llm_api_key="sk-...",

    # Embedding
    embedding_enabled=True,
    embedding_model="all-MiniLM-L6-v2",
)

engine = Engram(config)
```

### Save and Load

```python
# Save indexed data to disk
engine.save("./my_project_data")

# Load it back later (avoids re-indexing)
engine.load("./my_project_data")
```

### Environment Variables

Configuration can also come from environment variables:

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DEEPSEEK_API_KEY` | DeepSeekBackend | API authentication |
| `OPENAI_API_KEY` | OpenAIBackend | API authentication |
| `ANTHROPIC_API_KEY` | AnthropicBackend | API authentication |
| `OLLAMA_HOST` | OllamaBackend | Server URL |
| `OLLAMA_MODEL` | OllamaBackend | Model name |

---

## API Reference

### Engram Engine

```python
engine = Engram(config=None)

# Ingestion
engine.ingest(filepath) → str                    # Returns doc_id
engine.ingest_batch(filepaths) → List[str]       # Returns list of doc_ids

# Retrieval
engine.retrieve(query, top_k=None) → List[ChunkResult]

# Generation
engine.generate(query, top_k=None, temperature=0.3) → Response

# Document management
engine.list_documents() → List[Dict]
engine.remove_document(doc_id) → bool
engine.get_concepts(doc_id) → List[Concept]
engine.get_graph(doc_id) → ConceptGraph

# Persistence
engine.save(path)
engine.load(path)

# Stats
engine.get_stats() → Dict
```

### ChunkResult

```python
result.chunk          # Chunk object with text, words, metadata
result.score          # Final combined score (0.0 to 1.0)
result.source         # "hash", "vector", "dual", or "expansion"
result.hash_hits      # Number of hash table matches
result.vector_sim     # Cosine similarity (if vector retrieval was used)
result.rerank_score   # Score after re-ranking
result.match_details  # Dict with detailed scoring breakdown
```

### Response

```python
response.text         # Generated answer text
response.sources      # List[ChunkResult] used as sources
response.confidence   # Overall confidence score (0.0 to 1.0)
response.evaluation   # Dict with grounding score, hallucination flags, etc.
response.model        # LLM model name used
response.prompt_tokens    # Token count of the prompt
response.response_tokens  # Token count of the response
```

### Concept

```python
concept.concept_id       # Unique identifier
concept.label            # Human-readable name
concept.score            # TF-IDF weighted score
concept.frequency        # Occurrence count
concept.chunk_ids        # List of chunk IDs where it appears
concept.primary_chunk_id # Highest-scoring chunk
concept.definition       # Extracted definition text (if found)
concept.tags             # ["D"] definition, ["H"] heading, ["B"] bold
```

---

## Performance

Engram is designed for speed at every layer:

| Operation | Typical Time | Complexity |
|-----------|-------------|------------|
| Hash lookup | < 0.01ms | O(1) |
| Vector search (1K chunks) | < 1ms | O(n) |
| Re-ranking (20 results) | < 0.1ms | O(n) |
| Concept extraction | 50-200ms | O(n * m) |
| Full indexing (100 chunks) | 200-500ms | O(n * m) |
| Full retrieval pipeline | 0.1-5ms | O(1) hash / O(n) vector |

Where n = number of chunks and m = average chunk size.

The hash-based retrieval path is constant-time regardless of corpus size, making it ideal for large document collections. Vector retrieval scales linearly but is only used when embeddings are enabled.

---

## Design Principles

1. **Local-first**: The core pipeline runs entirely on your machine. No data leaves your system unless you explicitly configure a remote LLM or embedding service.

2. **Zero dependencies at the core**: The hash-based retrieval, concept extraction, and text processing use only the Python standard library. Optional features (PDF parsing, embeddings, LLMs) have their own optional dependencies.

3. **Graceful degradation**: Every optional component has a fallback. No embeddings? Hash retrieval still works. No LLM? Retrieval still returns chunks. No sentence-transformers? TF-IDF embeddings are used instead.

4. **Deterministic and debuggable**: Every scoring decision is transparent. `match_details` on each result shows exactly why it was ranked where it was. No black-box neural reranking in the core path.

5. **Extensible at every layer**: Parsers, chunkers, backends, and retrievers all use abstract base classes. Register your own implementations via factories.

6. **Type-safe**: All data structures are defined as dataclasses with full type annotations. The codebase is designed to work well with IDE autocompletion and static type checkers.

7. **Memory-conscious**: Binary serialization with gzip compression keeps storage small. LRU caches prevent unbounded memory growth. Streaming-friendly design avoids loading entire documents into memory unnecessarily.

---

## License

MIT

---

