from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set


DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "academic": {
        "abstract", "introduction", "methodology", "conclusion", "references",
        "hypothesis", "experiment", "findings", "analysis", "literature",
        "review", "cite", "citation", "journal", "proceedings", "peer",
        "reviewed", "doi", "arxiv", "et al", "ibid", "thesis", "dissertation",
        "bibliography", "appendix", "university", "professor", "research",
    },
    "technical": {
        "algorithm", "implementation", "architecture", "framework", "protocol",
        "interface", "module", "component", "configuration", "deployment",
        "database", "server", "client", "api", "sdk", "runtime", "compiler",
        "debug", "deploy", "docker", "kubernetes", "microservice", "pipeline",
        "repository", "version", "commit", "branch", "merge", "refactor",
    },
    "business": {
        "revenue", "profit", "market", "strategy", "stakeholder", "budget",
        "quarterly", "fiscal", "roi", "kpi", "growth", "acquisition",
        "merger", "valuation", "shareholder", "dividend", "portfolio",
        "compliance", "regulation", "audit", "inventory", "supply",
        "chain", "logistics", "procurement", "sustainability",
    },
    "legal": {
        "whereas", "hereinafter", "party", "parties", "agreement", "contract",
        "clause", "provision", "statute", "jurisdiction", "liability",
        "indemnification", "arbitration", "litigation", "plaintiff",
        "defendant", "court", "ruling", "verdict", "testimony", "evidence",
        "subpoena", "deposition", "attorney", "counsel", "bar",
    },
    "medical": {
        "patient", "diagnosis", "treatment", "symptom", "clinical",
        "trial", "pharmaceutical", "dosage", "therapeutic", "pathology",
        "anatomy", "physiology", "surgery", "prescription", "vaccine",
        "immunology", "cardiology", "oncology", "neurology", "radiology",
        "lab", "specimen", "biopsy", "prognosis", "mortality",
    },
    "science": {
        "hypothesis", "experiment", "observation", "variable", "control",
        "sample", "data", "measurement", "theory", "model", "simulation",
        "calibration", "spectrum", "molecule", "atom", "electron",
        "proton", "neutron", "genome", "species", "evolution", "ecology",
        "climate", "geology", "physics", "chemistry", "biology",
    },
    "self_help": {
        "goal", "habit", "mindset", "motivation", "success", "failure",
        "growth", "routine", "discipline", "gratitude", "meditation",
        "wellness", "balance", "purpose", "vision", "manifest",
        "affirmation", "journal", "reflection", "accountability",
    },
    "history": {
        "century", "dynasty", "empire", "revolution", "war", "treaty",
        "civilization", "era", "period", "ancient", "medieval", "colonial",
        "independence", "republic", "monarchy", "conquest", "migration",
        "artifact", "archaeology", "chronicle", "heritage", "tradition",
    },
}

HEADING_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(?:\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{3,60})", re.MULTILINE),
    re.compile(r"^\s*(?:Chapter|Section|Part|Lesson)\s+\d+[.:]\s*([A-Z][^\n]{3,60})", re.MULTILINE),
    re.compile(r"^\s*([A-Z][A-Z\s]{3,60})\s*$", re.MULTILINE),
    re.compile(r"^#\s+(.+)$", re.MULTILINE),
    re.compile(r"^##\s+(.+)$", re.MULTILINE),
    re.compile(r"^###\s+(.+)$", re.MULTILINE),
]

TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^#\s+(.+)$", re.MULTILINE),
    re.compile(r"^\s*(?:Chapter|Section|Part)\s+1[:\s]+(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*([A-Z][A-Z\s]{5,60})\s*$", re.MULTILINE),
    re.compile(r"^\s*([A-Z][^\n]{5,80})\s*$", re.MULTILINE),
]

LANGUAGE_INDICATORS: Dict[str, Set[str]] = {
    "en": {"the", "is", "are", "was", "were", "have", "has", "had", "been", "will", "would", "could", "should", "this", "that", "with", "from", "they", "their", "about"},
    "es": {"el", "la", "los", "las", "del", "por", "para", "como", "pero", "muy", "estos", "estas", "tiene", "puede", "desde"},
    "fr": {"les", "des", "une", "dans", "pour", "avec", "est", "sont", "pas", "sur", "mais", "nous", "vous", "leur", "cette"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "auf", "mit", "nicht", "auch", "aber", "werden", "haben", "kann"},
    "pt": {"que", "nao", "como", "para", "com", "uma", "por", "mais", "tem", "sao", "isso", "esta", "pode", "muito", "foi"},
    "it": {"che", "per", "con", "non", "una", "sono", "del", "della", "anche", "come", "piu", "questo", "stato", "dopo", "molto"},
    "nl": {"van", "het", "een", "en", "dat", "niet", "zijn", "voor", "ook", "maar", "dan", "bij", "nog", "wel", "veel"},
    "ru": {"и", "в", "не", "на", "что", "он", "как", "это", "по", "но", "из", "за", "его", "она", "быть"},
    "zh": {"的", "是", "了", "在", "不", "我", "有", "这", "他", "她", "它", "们", "就", "也", "都"},
    "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "する"},
    "ko": {"은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "에서", "와", "과", "한", "하다"},
    "ar": {"في", "من", "على", "إلى", "أن", "هذا", "التي", "الذي", "كان", "هو", "هي", "مع", "كل", "وقد", "كما"},
}


class MetadataDetector:

    def detect(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        title = self._detect_title(text, filename)
        if title:
            metadata["title"] = title

        language = self._detect_language(text)
        metadata["language"] = language

        domain = self._detect_domain(text)
        metadata["domain"] = domain

        headings = self._detect_headings(text)
        if headings:
            metadata["headings"] = headings[:50]
            metadata["heading_count"] = len(headings)

        structure = self._analyze_structure(text)
        metadata.update(structure)

        return metadata

    def _detect_title(self, text: str, filename: Optional[str] = None) -> str:
        if filename:
            name = os.path.splitext(filename)[0]
            name = re.sub(r"[-_]+", " ", name)
            name = re.sub(r"\s+", " ", name).strip()
            if len(name) > 3:
                return name

        lines = text.strip().split("\n")

        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                if len(title) > 2:
                    return title
            if len(line) > 5 and len(line) < 100:
                if line.isupper() and len(line.split()) > 1:
                    return line.title()
                if re.match(r"^[A-Z]", line) and not line.endswith("."):
                    words = line.split()
                    if 2 <= len(words) <= 12:
                        return line

        for pattern in TITLE_PATTERNS:
            match = pattern.search(text[:2000])
            if match:
                candidate = match.group(1).strip()
                if 3 < len(candidate) < 100:
                    return candidate

        return ""

    def _detect_language(self, text: str) -> str:
        sample = text[:5000].lower()
        sample = re.sub(r"[^a-z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff\s]", "", sample)

        cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", sample))
        total_chars = len(sample)
        if total_chars > 0 and cjk_chars / total_chars > 0.2:
            cjk_sample = re.findall(r"[\u4e00-\u9fff]", sample)
            jp_sample = re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", sample)
            ko_sample = re.findall(r"[\uac00-\ud7af]", sample)
            if len(jp_sample) > len(cjk_sample) * 0.3:
                return "ja"
            if len(ko_sample) > len(cjk_sample) * 0.3:
                return "ko"
            return "zh"

        arabic_chars = len(re.findall(r"[\u0600-\u06ff]", sample))
        if total_chars > 0 and arabic_chars / total_chars > 0.1:
            return "ar"

        cyrillic_chars = len(re.findall(r"[\u0400-\u04ff]", sample))
        if total_chars > 0 and cyrillic_chars / total_chars > 0.1:
            return "ru"

        words = set(re.findall(r"[a-z]{2,}", sample))
        if not words:
            return "en"

        scores: Dict[str, float] = {}
        for lang, indicators in LANGUAGE_INDICATORS.items():
            if lang in ("zh", "ja", "ko", "ar", "ru"):
                continue
            overlap = len(words & indicators)
            scores[lang] = overlap / max(len(indicators), 1)

        if not scores:
            return "en"

        best = max(scores, key=lambda k: scores[k])
        if scores[best] < 0.02:
            return "en"
        return best

    def _detect_domain(self, text: str) -> str:
        sample = text[:20000].lower()
        words = set(re.findall(r"[a-z]{3,}", sample))

        scores: Dict[str, float] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            overlap = len(words & keywords)
            score = overlap / max(len(keywords), 1)
            scores[domain] = score

        if not scores:
            return "unknown"

        best = max(scores, key=lambda k: scores[k])
        if scores[best] < 0.03:
            return "unknown"
        return best

    def _detect_headings(self, text: str) -> List[str]:
        headings: List[str] = []
        seen: set = set()

        for pattern in HEADING_PATTERNS:
            for match in pattern.finditer(text[:50000]):
                heading = match.group(1).strip() if match.lastindex else match.group(0).strip()
                heading = re.sub(r"\s+", " ", heading)
                if heading and heading not in seen and len(heading) > 2 and len(heading) < 100:
                    seen.add(heading)
                    headings.append(heading)

        return headings

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        lines = text.split("\n")

        non_empty = [l for l in lines if l.strip()]
        result["line_count"] = len(lines)
        result["non_empty_lines"] = len(non_empty)
        result["char_count"] = len(text)
        result["word_count"] = len(text.split())

        paragraph_breaks = text.count("\n\n")
        result["paragraph_count"] = max(paragraph_breaks, 1)

        code_indicators = ["def ", "class ", "import ", "function ", "var ", "const ", "let ", "#include", "public ", "private "]
        code_count = sum(1 for indicator in code_indicators if indicator in text[:10000])
        result["likely_contains_code"] = code_count >= 2

        url_count = len(re.findall(r"https?://\S+", text))
        result["url_count"] = url_count

        ref_pattern = re.compile(r"$$\d+$$|$$\d{4}$$|et al\.|doi:|arxiv:", re.IGNORECASE)
        ref_count = len(ref_pattern.findall(text[:30000]))
        result["reference_indicators"] = ref_count
        result["likely_has_references"] = ref_count > 3

        return result