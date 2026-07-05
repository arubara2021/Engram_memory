from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.types import Chunk, Concept

from .hash_table import (
    COMMON_WORDS,
    NOISE_TERMS,
    extract_ngrams_from_tokens,
    is_url_noise,
    normalize_text,
    tokenize_for_concepts,
)

TECHNICAL_SUFFIXES = [
    "tion", "ment", "ness", "ity", "ism", "ive", "ory", "able", "ible",
    "ful", "ous", "ize", "ise", "ify", "ate", "ent", "ant", "ing",
    "net", "ary", "acy", "ics", "oid", "gram", "graph", "scope",
    "meter", "ology", "onomy", "pathy", "genic", "phile", "phobe",
    "form", "fold", "ward", "wise",
]

LEADING_ARTICLES = {"the", "a", "an"}

DEFINITION_PATTERNS = [
    re.compile(
        r"([A-Z][\w\s-]{2,30}?)\s+(?:is defined as|is a type of|is a method|is a technique|is a process|is the process|is the ability|is the sum|is the product|is the result|is the number|is the set|is the ratio|is the average|is a function|is a measure|is a metric)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][\w\s-]{2,30}?)\s+(?:refers to|denotes|represents|describes|computes|calculates|determines|measures|quantifies|estimates|predicts|classifies|generates|produces|transforms|maps|converts|encodes|decodes)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the\s+)?([A-Z][\w\s-]{2,30}?)\s+(?:consists of|involves|includes|comprises|contains|requires|uses|utilizes|employs|leverages|exploits|applies|implements|executes|performs)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:we define|let us define|define)\s+([A-Z][\w\s-]{2,30}?)(?:\s+as|\s+to be)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:known as|called|termed|referred to as)\s+([A-Z][\w\s-]{2,30}?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Definition|Theorem|Lemma|Proposition|Property|Corollary)"
        r"\s*[:\d.]*\s*[:.]?\s*([A-Z][\w\s-]{2,30}?)\s+(?:is|are|denotes|refers)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([a-z][\w-]{3,25}?)\s+(?:is defined as|is a type of|refers to|is the process)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([a-z][\w-]{3,25}?)\s+(?:is a|are a)\s+(?:method|technique|approach|"
        r"mechanism|process|function|operation|layer|network|model|algorithm|"
        r"architecture|component|module|unit|block|structure|pattern)",
        re.IGNORECASE,
    ),
]

HEADING_PATTERNS = [
    re.compile(r"^\s*(?:\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{3,60})", re.MULTILINE),
    re.compile(
        r"^\s*(?:Chapter|Section|Part)\s+\d+[.:]\s*([A-Z][^\n]{3,60})",
        re.MULTILINE,
    ),
    re.compile(r"^\s*([A-Z][A-Z\s]{3,60})\s*$", re.MULTILINE),
    re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),
]

TOC_DOTS_RE = re.compile(r"\.{3,}")
TOC_PAGE_NUM_RE = re.compile(r"\.{2,}\s*\d+")
TOC_CAPS_LINE_RE = re.compile(r"^[A-Z][A-Z\s&]{8,}$")


def is_technical_term(word: str) -> bool:
    if len(word) < 3:
        return False
    if word in COMMON_WORDS or word in NOISE_TERMS:
        return False
    for suffix in TECHNICAL_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return True
    consonant_count = sum(1 for c in word if c in "bcdfghjklmnpqrstvwxyz")
    if consonant_count / max(len(word), 1) > 0.65 and len(word) > 4:
        return True
    return False


def strip_articles(term: str) -> str:
    words = term.split()
    if not words:
        return term
    while words and words[0].lower() in LEADING_ARTICLES and len(words) > 1:
        words = words[1:]
    while words and words[-1].lower() in LEADING_ARTICLES and len(words) > 1:
        words = words[:-1]
    return " ".join(words)


def normalize_concept(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"[^\w\s-]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    term = strip_articles(term)
    return term


def singular_form(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        return word[:-1]
    return word


def concept_canonical_key(term: str) -> str:
    cleaned = strip_articles(term)
    words = cleaned.lower().split()
    result = []
    for w in words:
        if w in LEADING_ARTICLES:
            continue
        result.append(singular_form(w))
    return " ".join(result)


def is_toc_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if TOC_DOTS_RE.search(stripped):
        return True
    if TOC_PAGE_NUM_RE.search(stripped):
        return True
    return False


def filter_concept_noise(concept: Dict) -> bool:
    term = concept["term"]
    if is_url_noise(term):
        return False
    if term.strip() in NOISE_TERMS:
        return False
    words = term.split()
    if all(w in NOISE_TERMS or w in COMMON_WORDS for w in words):
        return False
    if any(c.isdigit() for c in term) and len(term.split()) <= 2:
        has_alpha = any(c.isalpha() for c in term)
        if not has_alpha:
            return False
    if len(term.strip()) < 2:
        return False

    content_words = [w for w in words if w not in COMMON_WORDS and w not in NOISE_TERMS]
    if len(content_words) == 0:
        return False
    return True


class ConceptExtractor:

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._max_concepts = (
            getattr(config, "max_concepts", 0) if config else 0
        )
        self._min_freq = (
            getattr(config, "concept_min_freq", 2) if config else 2
        )

    def _dynamic_max(self, chunks: list, full_text: str) -> int:
        word_count = len(full_text.split()) if full_text else 0
        unique_words = (
            len(set(full_text.lower().split())) if full_text else 0
        )
        chunk_count = len(chunks)

        base = max(15, min(150, word_count // 1000))

        if unique_words > 3000:
            base = int(base * 1.3)
        elif unique_words > 1500:
            base = int(base * 1.15)

        if chunk_count > 200:
            base = int(base * 1.2)
        elif chunk_count > 100:
            base = int(base * 1.1)

        return max(15, min(150, base))

    def extract(
        self, chunks: List[Chunk], full_text: str
    ) -> List[Concept]:
        clean_text = full_text

        toc_text = self._extract_toc_text(clean_text)
        toc_terms: Set[str] = set()
        if toc_text:
            toc_tokens = tokenize_for_concepts(toc_text)
            for t in toc_tokens:
                if t not in COMMON_WORDS and len(t) > 2:
                    toc_terms.add(t)
            for ng, _ in extract_ngrams_from_tokens(toc_tokens, 3):
                toc_terms.add(ng)

        term_freq: Dict[str, int] = defaultdict(int)
        term_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        bold_terms: Set[str] = set()
        definition_terms: Set[str] = set()
        heading_terms: Set[str] = set()
        abbreviations: Dict[str, str] = {}

        headings = self._detect_headings(clean_text)
        for heading in headings:
            tokens = tokenize_for_concepts(heading)
            for t in tokens:
                heading_terms.add(t)
            for ng, _ in extract_ngrams_from_tokens(tokens, 3):
                heading_terms.add(ng)

        for pattern in DEFINITION_PATTERNS:
            for match in pattern.finditer(clean_text):
                term = match.group(1).strip().lower()
                if len(term) > 2 and not is_url_noise(term):
                    normalized = normalize_concept(term)
                    if normalized and not is_url_noise(normalized):
                        definition_terms.add(normalized)

        bold_pattern = re.compile(
            r"\*\*(.+?)\*\*|__(.+?)__", re.MULTILINE
        )
        for match in bold_pattern.finditer(clean_text):
            found = (match.group(1) or match.group(2) or "").strip()
            if found and not is_url_noise(found):
                normalized = normalize_concept(found)
                if normalized and not is_url_noise(normalized):
                    bold_terms.add(normalized)

        non_toc_chunks = []
        toc_chunks = []
        for chunk in chunks:
            if self._is_toc_chunk(chunk):
                toc_chunks.append(chunk)
            else:
                non_toc_chunks.append(chunk)

        for chunk in non_toc_chunks:
            seen: Set[str] = set()
            for word in chunk.words:
                if len(word) > 1 and word not in NOISE_TERMS:
                    term_freq[word] += 1
                    if word not in seen:
                        term_to_chunks[word].add(chunk.chunk_id)
                        seen.add(word)

            chunk_ngrams = extract_ngrams_from_tokens(chunk.words, 3)
            for ng, _ in chunk_ngrams:
                if not is_url_noise(ng):
                    term_freq[ng] += 1
                    if ng not in seen:
                        term_to_chunks[ng].add(chunk.chunk_id)
                        seen.add(ng)

        for chunk in toc_chunks:
            seen: Set[str] = set()
            for word in chunk.words:
                if len(word) > 1 and word not in NOISE_TERMS:
                    if word not in toc_terms:
                        term_freq[word] += 1
                        if word not in seen:
                            term_to_chunks[word].add(chunk.chunk_id)
                            seen.add(word)

        # Compute PMI for multi-word terms
        word_doc_freq: Dict[str, int] = defaultdict(int)
        total_chunks = len(non_toc_chunks)
        for chunk in non_toc_chunks:
            seen_words: Set[str] = set()
            for w in chunk.words:
                if w not in NOISE_TERMS and len(w) > 1:
                    if w not in seen_words:
                        word_doc_freq[w] += 1
                        seen_words.add(w)

        def pmi_score(term: str) -> float:
            words = term.split()
            if len(words) < 2:
                return 1.0
            joint = term_freq.get(term, 0)
            if joint == 0 or total_chunks == 0:
                return 0.0
            joint_prob = joint / total_chunks
            word_probs = []
            for w in words:
                wf = word_doc_freq.get(w, 0)
                if wf == 0:
                    return 0.0
                word_probs.append(wf / total_chunks)
            independent_prob = 1.0
            for wp in word_probs:
                independent_prob *= wp
            if independent_prob == 0:
                return 0.0
            import math
            pmi = math.log2(joint_prob / independent_prob)
            return max(pmi, 0.0)

        def genericity_ratio(term: str) -> float:
            words = term.split()
            if not words:
                return 1.0
            common_count = sum(1 for w in words if w in COMMON_WORDS)
            return common_count / len(words)

        concept_candidates: Dict[str, Dict] = {}

        for term in term_freq:
            if is_url_noise(term):
                continue
            if term in NOISE_TERMS:
                continue
            words_in_term = term.split()
            if all(w in NOISE_TERMS for w in words_in_term):
                continue

            freq = term_freq[term]
            word_count = len(words_in_term)
            is_definition = term in definition_terms
            is_heading = term in heading_terms
            is_bold = term in bold_terms

            min_freq_for_term = self._min_freq
            if is_definition:
                min_freq_for_term = 1
            if is_heading:
                min_freq_for_term = 1

            if freq < min_freq_for_term and word_count == 1:
                if not is_definition and not is_heading:
                    continue
            if freq < min_freq_for_term and word_count > 1:
                if not is_definition and not is_heading:
                    continue

            score = 0.0

            freq_score = math.log(freq + 1) * 3
            score += min(freq_score, 20)

            chunk_count = len(term_to_chunks.get(term, set()))
            if chunk_count >= 2:
                score += min(chunk_count * 1.5, 15)

            if word_count >= 2:
                score += 10
            if word_count >= 3:
                score += 5

            if is_definition:
                score += 18
            if is_heading:
                score += 12
            if is_bold:
                score += 10

            specificity_bonus = 0
            if term not in COMMON_WORDS:
                specificity_bonus += 10
            if any(is_technical_term(w) for w in words_in_term):
                specificity_bonus += 5
            if word_count >= 2:
                specificity_bonus += 5
            score += specificity_bonus

            genericity_penalty = 0
            if word_count == 1 and term in COMMON_WORDS:
                genericity_penalty += 25
            if word_count == 1 and freq > len(chunks) * 0.7:
                genericity_penalty += 15
            score -= genericity_penalty

            score_threshold = 10 if is_definition else 12

            # PMI penalty for incoherent multi-word terms
            if word_count >= 2:
                pmi = pmi_score(term)
                if pmi < 0.5:
                    score *= max(0.2, pmi)
                elif pmi >= 2.0:
                    score *= 1.15

            # Genericity penalty: too many common words in the term
            gen_ratio = genericity_ratio(term)
            if gen_ratio >= 0.5 and word_count >= 2:
                score *= (1.0 - gen_ratio * 0.8)

            # Penalize terms where the content word count is low
            content_in_term = [w for w in words_in_term if w not in COMMON_WORDS and w not in NOISE_TERMS]
            if word_count >= 2 and len(content_in_term) < 2:
                score *= 0.3

            if score >= score_threshold:
                cleaned = normalize_concept(term)
                if not cleaned or len(cleaned) < 2:
                    continue

                chunk_ids = list(term_to_chunks.get(term, set()))
                primary = chunk_ids[0] if chunk_ids else None

                canonical = concept_canonical_key(cleaned)
                if canonical in concept_candidates:
                    existing = concept_candidates[canonical]
                    existing["score"] = max(existing["score"], score)
                    existing["frequency"] += freq
                    for cid in chunk_ids:
                        if cid not in existing["chunk_ids"]:
                            existing["chunk_ids"].append(cid)
                    if is_definition:
                        existing["in_definition"] = True
                    if is_heading:
                        existing["in_heading"] = True
                    if is_bold:
                        existing["in_bold"] = True
                    if len(cleaned) < len(existing["term"]):
                        existing["term"] = cleaned
                    continue

                if cleaned in abbreviations:
                    full_term = abbreviations[cleaned]
                    if full_term in concept_candidates:
                        concept_candidates[full_term]["score"] += 5
                        continue

                concept_candidates[canonical] = {
                    "term": cleaned,
                    "score": max(score, 0),
                    "frequency": freq,
                    "chunk_ids": chunk_ids[:15],
                    "primary_chunk_id": primary,
                    "word_count": word_count,
                    "in_definition": is_definition,
                    "in_heading": is_heading,
                    "in_bold": is_bold,
                }

        filtered = [
            c
            for c in concept_candidates.values()
            if filter_concept_noise(c)
        ]
        ranked = sorted(
            filtered, key=lambda x: (-x["score"], -x["frequency"])
        )

        effective_max = self._max_concepts
        if effective_max <= 0:
            effective_max = self._dynamic_max(chunks, full_text)
        top = ranked[:effective_max]

        concepts: List[Concept] = []
        for c in top:
            tags: List[str] = []
            if c.get("in_definition"):
                tags.append("D")
            if c.get("in_heading"):
                tags.append("H")
            if c.get("in_bold"):
                tags.append("B")

            concepts.append(
                Concept(
                    concept_id=concept_canonical_key(c["term"]),
                    label=c["term"],
                    ngrams=[c["term"]] + c["term"].split(),
                    frequency=c["frequency"],
                    chunk_ids=c["chunk_ids"],
                    score=c["score"],
                    tags=tags,
                    primary_chunk_id=c.get("primary_chunk_id"),
                    definition="",
                    in_definition=c.get("in_definition", False),
                    in_heading=c.get("in_heading", False),
                    in_bold=c.get("in_bold", False),
                    word_count=c.get("word_count", 1),
                )
            )

        return concepts

    def _detect_headings(self, text: str) -> List[str]:
        headings: List[str] = []
        seen: Set[str] = set()
        for line in text.split("\n"):
            line = line.strip()
            for pattern in HEADING_PATTERNS:
                match = pattern.match(line)
                if match:
                    found = (
                        match.group(1).strip()
                        if match.lastindex
                        else line
                    )
                    if (
                        not is_url_noise(found)
                        and found not in seen
                        and not is_toc_text(found)
                    ):
                        seen.add(found)
                        headings.append(found)
                    break
        return headings

    def _is_toc_chunk(self, chunk: Chunk) -> bool:
        text = chunk.text.strip()
        if not text:
            return False
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return False
        toc_count = sum(1 for l in lines if is_toc_text(l))
        return toc_count / len(lines) > 0.5

    def _extract_toc_text(self, text: str) -> str:
        toc_parts: List[str] = []
        in_toc = False
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if in_toc:
                    toc_parts.append(line)
                continue
            upper = stripped.upper()
            if (
                "TABLE OF CONTENTS" in upper
                or upper == "CONTENTS"
                or upper == "INDEX"
            ):
                in_toc = True
                continue
            if in_toc:
                if is_toc_text(stripped):
                    toc_parts.append(stripped)
                elif len(stripped) > 80:
                    in_toc = False
        return "\n".join(toc_parts)
