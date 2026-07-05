from __future__ import annotations

import pytest

from engram.core.types import Chunk, ChunkResult
from engram.generate.base import LLMBackend
from engram.generate.prompt import PromptBuilder
from engram.generate.evaluator import ResponseEvaluator
from engram.generate.factory import LLMFactory
from engram.generate.ollama import OllamaBackend


def _make_chunk_results() -> list:
    chunks_data = [
        ("c0", "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.", 0.9),
        ("c1", "Neural networks consist of layers of interconnected nodes called neurons. Each connection has a weight.", 0.8),
        ("c2", "Training involves forward propagation and backpropagation. The loss function measures prediction error.", 0.7),
    ]
    results = []
    for cid, text, score in chunks_data:
        words = text.lower().split()
        chunk = Chunk(
            chunk_id=cid, doc_id="d", text=text,
            word_count=len(words), page=1, section="Chapter 1",
            index=0, words=words,
        )
        results.append(ChunkResult(chunk=chunk, score=score, source="hash"))
    return results


class TestPromptBuilder:

    def test_build_rag_prompt(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_rag_prompt("What is machine learning?", chunks)

        assert "What is machine learning?" in prompt
        assert "Machine learning" in prompt
        assert "SOURCE TEXT" in prompt
        assert "ANSWER" in prompt

    def test_build_rag_prompt_with_system(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_rag_prompt(
            "What is ML?", chunks, system_prompt="Custom system prompt."
        )
        assert "Custom system prompt." in prompt

    def test_build_rag_prompt_empty_chunks(self):
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt("What is ML?", [])
        assert "No relevant source text" in prompt

    def test_build_no_context_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_no_context_prompt("What is quantum physics?")
        assert "quantum physics" in prompt
        assert "not covered" in prompt.lower()

    def test_build_evaluation_prompt(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_evaluation_prompt(
            "What is ML?", "ML is AI that learns.", chunks
        )
        assert "STUDENT" in prompt
        assert "EVALUATION" in prompt
        assert "What is ML?" in prompt

    def test_build_question_generation_prompt(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_question_generation_prompt(
            "machine learning", chunks, "understanding"
        )
        assert "machine learning" in prompt
        assert "understanding" in prompt
        assert "SOURCE TEXT" in prompt

    def test_build_summarization_prompt(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_summarization_prompt(chunks, max_length=100)
        assert "Summarize" in prompt
        assert "100" in prompt

    def test_source_headers_include_info(self):
        builder = PromptBuilder()
        chunks = _make_chunk_results()
        prompt = builder.build_rag_prompt("test", chunks)
        assert "(relevance:" in prompt


class TestResponseEvaluator:

    def test_evaluate_grounded(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        response = "Machine learning is a subset of artificial intelligence that learns from data."
        result = evaluator.evaluate(response, chunks)

        assert result["grounding_score"] > 0
        assert result["is_grounded"] is True

    def test_evaluate_ungrounded(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        response = "Quantum entanglement allows instantaneous teleportation across galactic wormholes."
        result = evaluator.evaluate(response, chunks)

        assert result["grounding_score"] < 0.8

    def test_evaluate_empty_response(self):
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate("", _make_chunk_results())
        assert result["grounding_score"] == 0.0
        assert result["is_grounded"] is False

    def test_evaluate_empty_chunks(self):
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate("Some response", [])
        assert result["grounding_score"] == 0.0

    def test_compute_confidence(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        conf = evaluator.compute_confidence(chunks, final_top_k=5)
        assert 0 < conf <= 1.0

    def test_compute_confidence_empty(self):
        evaluator = ResponseEvaluator()
        conf = evaluator.compute_confidence([], final_top_k=5)
        assert conf == 0.0

    def test_check_source_grounding(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        result = evaluator.check_source_grounding(
            "Machine learning is AI that learns from data", chunks
        )
        assert result["grounded"] is True
        assert result["best_overlap"] > 0

    def test_check_source_grounding_unsupported(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        result = evaluator.check_source_grounding(
            "Quantum entanglement allows teleportation", chunks
        )
        assert result["grounded"] is False

    def test_detect_uncertainty(self):
        evaluator = ResponseEvaluator()
        chunks = _make_chunk_results()
        response = "This information is not in the provided documents."
        result = evaluator.evaluate(response, chunks)
        assert result["has_uncertainty_language"] is True


class TestLLMFactory:

    def test_list_backends(self):
        backends = LLMFactory.list_backends()
        assert "deepseek" in backends
        assert "openai" in backends
        assert "ollama" in backends
        assert "anthropic" in backends

    def test_create_deepseek(self):
        from engram.generate.deepseek import DeepSeekBackend
        backend = LLMFactory.create("deepseek")
        assert isinstance(backend, DeepSeekBackend)

    def test_create_openai(self):
        from engram.generate.openai import OpenAIBackend
        backend = LLMFactory.create("openai")
        assert isinstance(backend, OpenAIBackend)

    def test_create_ollama(self):
        from engram.generate.ollama import OllamaBackend
        backend = LLMFactory.create("ollama")
        assert isinstance(backend, OllamaBackend)

    def test_create_anthropic(self):
        from engram.generate.anthropic import AnthropicBackend
        backend = LLMFactory.create("anthropic")
        assert isinstance(backend, AnthropicBackend)

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            LLMFactory.create("nonexistent")

    def test_register_custom(self):
        class CustomBackend(LLMBackend):
            def __init__(self, config=None):
                pass
            def generate(self, prompt, temperature=0.3, max_tokens=2048):
                return "custom response"
            def is_available(self):
                return True

        LLMFactory.register("custom", CustomBackend)
        backend = LLMFactory.create("custom")
        assert backend.generate("test") == "custom response"

    def test_get_available(self):
        available = LLMFactory.get_available()
        assert isinstance(available, list)