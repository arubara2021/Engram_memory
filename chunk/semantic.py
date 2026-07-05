from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import BaseChunker
from .overlap import OverlapHandler
from .reference import ReferenceStripper
from ..core.types import Chunk


class SemanticChunker(BaseChunker):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._overlap = OverlapHandler()
        self._ref_stripper = ReferenceStripper()
        self._threshold = getattr(config, "semantic_threshold", 0.5) if config else 0.5
        self._strip_refs = getattr(config, "strip_references", True) if config else True
        self._overlap_sentences = getattr(config, "chunk_overlap_sentences", 2) if config else 2
        self._embedder = None
        self._embedder_loaded = False

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if self._strip_refs:
            text = self._ref_stripper.strip(text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            if sentences:
                return [self._make_chunk(sentences[0], "chunk_0", doc_id, "", 0)]
            return []

        embeddings = self._embed_sentences(sentences)

        if embeddings is not None:
            breakpoints = self._find_breakpoints(sentences, embeddings)
            segments = self._split_at_breakpoints(sentences, breakpoints)
        else:
            segments = self._fallback_split(sentences)

        chunks: List[Chunk] = []
        idx = 0

        for segment_sents in segments:
            segment_text = " ".join(segment_sents)
            segment_wc = self._word_count(segment_text)

            if segment_wc > self._max_words:
                sub_chunks = self._force_split(segment_text, doc_id, idx)
                chunks.extend(sub_chunks)
                idx += len(sub_chunks)
            elif segment_wc >= self._min_words:
                chunks.append(self._make_chunk(
                    segment_text, f"chunk_{idx}", doc_id, "", idx
                ))
                idx += 1
            else:
                if chunks:
                    last = chunks[-1]
                    combined = last.text + " " + segment_text
                    if self._word_count(combined) <= self._max_words:
                        chunks[-1] = self._make_chunk(
                            combined, last.chunk_id, doc_id, "", last.index
                        )
                    else:
                        chunks.append(self._make_chunk(
                            segment_text, f"chunk_{idx}", doc_id, "", idx
                        ))
                        idx += 1
                else:
                    chunks.append(self._make_chunk(
                        segment_text, f"chunk_{idx}", doc_id, "", idx
                    ))
                    idx += 1

        if self._overlap_sentences > 0 and len(chunks) > 1:
            chunks = self._overlap.apply(chunks, self._overlap_sentences)

        return chunks

    def _load_embedder(self) -> bool:
        if self._embedder_loaded:
            return self._embedder is not None
        self._embedder_loaded = True

        try:
            from sentence_transformers import SentenceTransformer

            model_name = "all-MiniLM-L6-v2"
            if self.config:
                model_name = getattr(self.config, "embedding_model", model_name)
            self._embedder = SentenceTransformer(model_name)
            return True
        except ImportError:
            try:
                import numpy as np

                self._embedder = "tfidf"
                return True
            except ImportError:
                self._embedder = None
                return False

    def _embed_sentences(self, sentences: List[str]):
        if not self._load_embedder():
            return None

        try:
            if self._embedder == "tfidf":
                return self._tfidf_embed(sentences)

            vectors = self._embedder.encode(
                sentences, show_progress_bar=False, normalize_embeddings=True
            )
            return vectors
        except Exception:
            return None

    def _tfidf_embed(self, sentences: List[str]):
        import math

        tokenized = []
        for s in sentences:
            tokens = self._tokenize(s)
            tokenized.append(tokens)

        df: Dict[str, int] = {}
        for tokens in tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        n = len(sentences)
        vocab = sorted(df.keys())
        vocab_idx = {w: i for i, w in enumerate(vocab)}

        vectors = []
        for tokens in tokenized:
            vec = [0.0] * len(vocab)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                if t in vocab_idx:
                    idf = math.log((n + 1) / (df[t] + 1)) + 1
                    vec[vocab_idx[t]] = count * idf
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return vectors

    def _cosine_sim(self, a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _find_breakpoints(self, sentences: List[str], embeddings) -> List[int]:
        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_sim(
                list(embeddings[i]), list(embeddings[i + 1])
            )
            similarities.append(sim)

        if not similarities:
            return []

        sorted_sims = sorted(similarities)
        percentile_idx = int(len(sorted_sims) * self._threshold)
        percentile_idx = max(0, min(percentile_idx, len(sorted_sims) - 1))
        threshold_value = sorted_sims[percentile_idx]

        breakpoints: List[int] = []
        for i, sim in enumerate(similarities):
            if sim < threshold_value:
                breakpoints.append(i + 1)

        return breakpoints

    def _split_at_breakpoints(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[List[str]]:
        if not breakpoints:
            return [sentences]

        segments: List[List[str]] = []
        start = 0
        for bp in breakpoints:
            segment = sentences[start:bp]
            if segment:
                segments.append(segment)
            start = bp
        if start < len(sentences):
            segments.append(sentences[start:])

        return segments

    def _fallback_split(self, sentences: List[str]) -> List[List[str]]:
        segments: List[List[str]] = []
        current: List[str] = []
        current_wc = 0

        for sent in sentences:
            sent_wc = self._word_count(sent)
            if current_wc + sent_wc > self._max_words and current:
                segments.append(current)
                current = []
                current_wc = 0
            current.append(sent)
            current_wc += sent_wc

        if current:
            segments.append(current)

        return segments if segments else [sentences]

    def _force_split(self, text: str, doc_id: str, start_idx: int) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        current_text = ""
        current_wc = 0
        idx = start_idx

        for sent in sentences:
            sent_wc = self._word_count(sent)
            if current_wc + sent_wc > self._max_words and current_text.strip():
                if current_wc >= self._min_words:
                    chunks.append(self._make_chunk(
                        current_text.strip(), f"chunk_{idx}", doc_id, "", idx
                    ))
                    idx += 1
                current_text = ""
                current_wc = 0
            if current_text:
                current_text += " " + sent
            else:
                current_text = sent
            current_wc += sent_wc

        if current_text.strip() and current_wc >= self._min_words:
            chunks.append(self._make_chunk(
                current_text.strip(), f"chunk_{idx}", doc_id, "", idx
            ))

        return chunks