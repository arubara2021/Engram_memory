from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.types import ChunkResult


class PromptBuilder:

    def __init__(self) -> None:
        self._default_system = (
            "You are a helpful assistant that answers questions based ONLY on "
            "the provided source text. If the answer is not found in the source "
            "text, say 'This information is not in the provided documents.' "
            "Do not use any outside knowledge. Always cite which source you used."
        )

    def build_rag_prompt(
        self,
        question: str,
        chunks: List[ChunkResult],
        system_prompt: Optional[str] = None,
    ) -> str:
        system = system_prompt or self._default_system
        parts = [system, ""]

        parts.append("SOURCE TEXT:")
        parts.append("")

        if not chunks:
            parts.append("[No relevant source text found]")
        else:
            for i, cr in enumerate(chunks, 1):
                source_info = self._format_source_header(i, cr)
                parts.append(source_info)
                parts.append(cr.chunk.text)
                parts.append("")

        parts.append("---")
        parts.append("")
        parts.append(f"QUESTION: {question}")
        parts.append("")
        parts.append("ANSWER:")

        return "\n".join(parts)

    def build_no_context_prompt(self, question: str) -> str:
        return (
            "The user asked a question but no relevant source text was found "
            "in the provided documents.\n\n"
            f"QUESTION: {question}\n\n"
            "Please respond with: 'This topic is not covered in the documents "
            "you provided. I can only answer questions based on your uploaded "
            "materials.'"
        )

    def build_evaluation_prompt(
        self,
        question: str,
        student_answer: str,
        chunks: List[ChunkResult],
    ) -> str:
        parts = [
            "You are an answer evaluator. Compare the student's answer against "
            "the source text. Identify what they got right, what they missed, "
            "and what they included that is not in the source.",
            "",
            "SOURCE TEXT:",
            "",
        ]

        for i, cr in enumerate(chunks, 1):
            parts.append(f"[{i}] (relevance: {cr.score:.2f}) {cr.chunk.text}")
            parts.append("")

        parts.append("---")
        parts.append("")
        parts.append(f"QUESTION: {question}")
        parts.append("")
        parts.append(f"STUDENT'S ANSWER: {student_answer}")
        parts.append("")
        parts.append(
            "EVALUATION FORMAT:\n"
            "CORRECT: [list what the student got right from the source]\n"
            "MISSED: [list important points from the source the student omitted]\n"
            "UNVERIFIABLE: [list claims the student made that are not in the source]\n"
            "FEEDBACK: [specific, actionable feedback]"
        )

        return "\n".join(parts)

    def build_question_generation_prompt(
        self,
        concept: str,
        chunks: List[ChunkResult],
        difficulty: str = "understanding",
    ) -> str:
        difficulty_instructions = {
            "recall": (
                "Generate a RECALL question that tests whether the student can "
                "remember specific facts from their notes. The answer should be "
                "found directly in the source text. "
                "Example format: 'According to your notes, what is...?'"
            ),
            "understanding": (
                "Generate an UNDERSTANDING question that tests whether the student "
                "can explain a concept in their own words. The answer should require "
                "connecting information from the source text. "
                "Example format: 'Explain how... as described in your notes.'"
            ),
            "application": (
                "Generate an APPLICATION question that tests whether the student "
                "can apply the concept to a new scenario. The answer should require "
                "using knowledge from the source text in a new context. "
                "Example format: 'Based on what your notes describe, what would happen if...?'"
            ),
        }

        instruction = difficulty_instructions.get(difficulty, difficulty_instructions["understanding"])

        parts = [
            "You are a question generator. Create a question about the given concept "
            "using ONLY the provided source text. Do not add any information not in "
            "the source.",
            "",
            f"CONCEPT: {concept}",
            f"DIFFICULTY: {difficulty}",
            f"INSTRUCTION: {instruction}",
            "",
            "SOURCE TEXT:",
            "",
        ]

        for i, cr in enumerate(chunks, 1):
            parts.append(f"[{i}] {cr.chunk.text}")
            parts.append("")

        parts.append("---")
        parts.append("")
        parts.append(
            "Generate the question and provide the expected answer based ONLY on the source text.\n"
            "FORMAT:\n"
            "QUESTION: [your question]\n"
            "EXPECTED ANSWER: [the answer based on the source text]\n"
            "SOURCE REFERENCE: [which source chunk(s) the answer comes from]"
        )

        return "\n".join(parts)

    def build_summarization_prompt(
        self,
        chunks: List[ChunkResult],
        max_length: Optional[int] = None,
    ) -> str:
        parts = [
            "Summarize the following source text. "
            "Stay strictly within the information provided. "
            "Do not add external knowledge.",
            "",
        ]

        if max_length:
            parts.append(f"Keep the summary under {max_length} words.")
            parts.append("")

        parts.append("SOURCE TEXT:")
        parts.append("")

        for i, cr in enumerate(chunks, 1):
            parts.append(f"[{i}] {cr.chunk.text}")
            parts.append("")

        parts.append("---")
        parts.append("")
        parts.append("SUMMARY:")

        return "\n".join(parts)

    def _format_source_header(self, index: int, cr: ChunkResult) -> str:
        parts = [f"[{index}]"]
        if cr.score > 0:
            parts.append(f"(relevance: {cr.score:.2f})")
        if cr.chunk.section:
            parts.append(f"section: {cr.chunk.section}")
        if cr.chunk.page and cr.chunk.page > 0:
            parts.append(f"page: {cr.chunk.page}")
        return " ".join(parts)