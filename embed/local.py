from __future__ import annotations

import math
from typing import Any, List, Optional


class LocalEmbedder:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self._model: Any = None
        self._is_loaded = False
        self._backend: Optional[str] = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        if self._is_loaded:
            return True
        return self._detect_backend()

    def load(self) -> bool:
        if self._is_loaded:
            return True

        if self._load_sentence_transformers():
            return True
        if self._load_onnx():
            return True
        if self._load_tfidf():
            return True

        return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._is_loaded:
            if not self.load():
                return self._fallback_embed(texts)

        if self._backend == "sentence_transformers":
            return self._embed_st(texts)
        if self._backend == "onnx":
            return self._embed_onnx(texts)
        if self._backend == "tfidf":
            return self._embed_tfidf(texts)

        return self._fallback_embed(texts)

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _detect_backend(self) -> bool:
        try:
            import sentence_transformers

            return True
        except ImportError:
            pass
        try:
            import onnxruntime

            return True
        except ImportError:
            pass
        try:
            import numpy

            return True
        except ImportError:
            pass
        return False

    def _load_sentence_transformers(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_embedding_dimension()
            self._backend = "sentence_transformers"
            self._is_loaded = True
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _load_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            import numpy as np

            try:
                session = ort.InferenceSession(
                    f"{self.model_name}.onnx",
                    providers=["CPUExecutionProvider"],
                )
                self._model = session
                self._backend = "onnx"
                self._is_loaded = True
                return True
            except Exception:
                return False
        except ImportError:
            return False

    def _load_tfidf(self) -> bool:
        try:
            import numpy

            self._backend = "tfidf"
            self._model = {}
            self._is_loaded = True
            return True
        except ImportError:
            return False

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self._model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32,
            )
            return [v.tolist() for v in vectors]
        except Exception:
            return self._fallback_embed(texts)

    def _embed_onnx(self, texts: List[str]) -> List[List[float]]:
        try:
            import numpy as np

            results: List[List[float]] = []
            for text in texts:
                tokens = self._tokenize_simple(text)
                input_ids = [hash(t) % 30000 for t in tokens[:512]]
                attention_mask = [1] * len(input_ids)
                while len(input_ids) < 128:
                    input_ids.append(0)
                    attention_mask.append(0)
                input_ids = input_ids[:128]
                attention_mask = attention_mask[:128]

                inputs = {
                    "input_ids": np.array([input_ids], dtype=np.int64),
                    "attention_mask": np.array([attention_mask], dtype=np.int64),
                }

                outputs = self._model.run(None, inputs)
                hidden = outputs[0][0]
                mask = np.array(attention_mask, dtype=np.float32)
                mask = mask / max(mask.sum(), 1)
                pooled = (hidden * mask[:, np.newaxis]).sum(axis=0)
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
                results.append(pooled.tolist()[: self._dimension])

            return results
        except Exception:
            return self._fallback_embed(texts)

    def _embed_tfidf(self, texts: List[str]) -> List[List[float]]:
        import math

        all_tokens = [self._tokenize_simple(t) for t in texts]

        df: dict = {}
        for tokens in all_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        n = len(texts)
        vocab = sorted(df.keys())[:10000]
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        dim = min(len(vocab), self._dimension)

        results: List[List[float]] = []
        for tokens in all_tokens:
            vec = [0.0] * dim
            tf: dict = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                if t in vocab_idx:
                    idx = vocab_idx[t] % dim
                    idf = math.log((n + 1) / (df[t] + 1)) + 1
                    vec[idx] += count * idf
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)

        self._dimension = dim
        return results

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        return self._embed_tfidf(texts)

    def _tokenize_simple(self, text: str) -> List[str]:
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return [t for t in text.split() if len(t) > 1]