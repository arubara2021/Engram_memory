from __future__ import annotations

from typing import Dict, List, Optional, Set


DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "academic": {
        "abstract", "introduction", "methodology", "conclusion", "references",
        "hypothesis", "experiment", "findings", "analysis", "literature",
        "review", "cite", "citation", "journal", "proceedings", "peer",
        "reviewed", "doi", "arxiv", "thesis", "dissertation",
        "bibliography", "appendix", "university", "professor", "research",
    },
    "technical": {
        "algorithm", "implementation", "architecture", "framework", "protocol",
        "interface", "module", "component", "configuration", "deployment",
        "database", "server", "client", "runtime", "compiler",
        "debug", "deploy", "pipeline", "repository", "version",
        "commit", "branch", "merge", "refactor", "optimize",
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
        "subpoena", "deposition", "attorney", "counsel",
    },
    "medical": {
        "patient", "diagnosis", "treatment", "symptom", "clinical",
        "trial", "pharmaceutical", "dosage", "therapeutic", "pathology",
        "anatomy", "physiology", "surgery", "prescription", "vaccine",
        "immunology", "cardiology", "oncology", "neurology", "radiology",
        "biopsy", "prognosis", "mortality", "morbidity",
    },
    "science": {
        "hypothesis", "experiment", "observation", "variable", "control",
        "sample", "measurement", "theory", "model", "simulation",
        "calibration", "spectrum", "molecule", "atom", "electron",
        "genome", "species", "evolution", "ecology", "climate",
        "geology", "physics", "chemistry", "biology", "quantum",
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
    "code": {
        "function", "class", "method", "variable", "import", "module",
        "package", "library", "dependency", "compile", "runtime",
        "debug", "test", "deploy", "build", "refactor", "iterate",
        "async", "await", "promise", "callback", "closure", "interface",
        "abstract", "inheritance", "polymorphism", "encapsulation",
    },
    "finance": {
        "stock", "bond", "equity", "debt", "interest", "rate",
        "inflation", "deflation", "currency", "exchange", "market",
        "index", "fund", "asset", "liability", "balance", "sheet",
        "income", "statement", "cash", "flow", "margin", "yield",
    },
}


def get_domain_keywords(domain: str) -> Set[str]:
    return DOMAIN_KEYWORDS.get(domain.lower(), set())


def detect_domain(text: str, sample_size: int = 20000) -> str:
    import re

    sample = text[:sample_size].lower()
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


def get_all_domains() -> List[str]:
    return sorted(DOMAIN_KEYWORDS.keys())


def get_domain_overlap(text: str, domain: str) -> float:
    import re

    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    keywords = get_domain_keywords(domain)
    if not keywords:
        return 0.0
    return len(words & keywords) / len(keywords)