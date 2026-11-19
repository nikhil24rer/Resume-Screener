"""Rule-based skill taxonomy, extraction and (separately labelled) inference.

Design choice: a curated taxonomy + word-boundary regex is transparent and
explainable (every extracted skill can be traced to a literal string in the
resume).  Its weakness is coverage: skills outside the taxonomy are invisible.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

# canonical skill -> list of aliases (the canonical name is always an alias too)
SKILL_TAXONOMY: Dict[str, List[str]] = {
    # languages
    "python": [], "java": [], "javascript": ["js"], "typescript": [],
    "c++": [], "c#": [], "go": ["golang"], "sql": [], "r": ["r programming"],
    "scala": [], "bash": ["shell scripting", "shell script"],
    # data
    "pandas": [], "numpy": [], "scipy": [], "matplotlib": [], "seaborn": [],
    "power bi": ["powerbi"], "tableau": [], "excel": ["ms excel", "microsoft excel"],
    "data analysis": ["data analytics", "exploratory data analysis", "eda"],
    "data visualization": ["data visualisation", "dashboards"],
    "etl": ["data pipelines", "data pipeline"], "spark": ["pyspark", "apache spark"],
    "hadoop": [], "airflow": ["apache airflow"],
    "statistics": ["statistical analysis", "statistical modeling"],
    # ML / NLP
    "machine learning": ["ml"], "deep learning": [],
    "nlp": ["natural language processing"], "computer vision": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "tensorflow": [], "keras": [], "pytorch": [], "xgboost": [],
    "tf-idf": ["tfidf", "tf idf"], "transformers": ["bert", "huggingface transformers"],
    "hugging face": ["huggingface"], "opencv": [], "feature engineering": [],
    "mlops": [], "model deployment": ["deployed models", "model serving"],
    "time series": ["time-series", "forecasting"],
    "recommendation systems": ["recommender systems", "recommendation engine"],
    "llm": ["llms", "large language models"],
    # web
    "flask": [], "django": [], "fastapi": [], "streamlit": [],
    "react": ["react.js", "reactjs"],
    "node.js": ["nodejs", "node js", "express.js"], "html": ["html5"], "css": ["css3"],
    "rest api": ["rest apis", "restful", "restful apis", "rest"],
    # databases
    "mysql": [], "postgresql": ["postgres"], "mongodb": [], "redis": [], "nosql": [],
    # devops / tools
    "docker": [], "kubernetes": ["k8s"], "aws": ["amazon web services"],
    "azure": [], "gcp": ["google cloud"], "git": [], "github": [],
    "ci/cd": ["cicd", "ci cd"], "jenkins": [], "linux": [], "terraform": [],
    "agile": ["scrum"], "unit testing": ["pytest", "unittest"],
}

# If a resume shows skill A it *probably* also covers B.  These are INFERENCES:
# they are reported separately and never counted as confirmed matches.
IMPLIED_SKILLS: Dict[str, List[str]] = {
    "pandas": ["python"], "numpy": ["python"], "scipy": ["python"],
    "scikit-learn": ["machine learning", "python"], "xgboost": ["machine learning"],
    "tensorflow": ["deep learning", "machine learning"], "keras": ["deep learning"],
    "pytorch": ["deep learning", "machine learning"],
    "deep learning": ["machine learning"],
    "transformers": ["nlp", "deep learning"], "hugging face": ["nlp"],
    "tf-idf": ["nlp"], "llm": ["nlp"], "opencv": ["computer vision"],
    "flask": ["python", "rest api"], "django": ["python"],
    "fastapi": ["python", "rest api"], "streamlit": ["python"],
    "mysql": ["sql"], "postgresql": ["sql"], "matplotlib": ["data visualization"],
    "seaborn": ["data visualization"], "power bi": ["data visualization"],
    "tableau": ["data visualization"], "github": ["git"], "kubernetes": ["docker"],
    "react": ["javascript", "html", "css"], "node.js": ["javascript"],
    "typescript": ["javascript"], "mlops": ["model deployment"],
}


def _build_regex():
    alias_to_canon: Dict[str, str] = {}
    for canon, aliases in SKILL_TAXONOMY.items():
        for a in [canon] + aliases:
            alias_to_canon[a.lower()] = canon
    ordered = sorted(alias_to_canon, key=len, reverse=True)  # longest match first
    body = "|".join(re.escape(a) for a in ordered)
    # Custom boundaries so c++, c#, node.js, ci/cd work and "java" != "javascript".
    rx = re.compile(rf"(?<![\w+#])({body})(?![\w+#])", re.IGNORECASE)
    return rx, alias_to_canon


_SKILL_RX, _ALIAS_TO_CANON = _build_regex()


def canonical_token(skill: str) -> str:
    """Single-token form of a skill so multi-word skills become one TF-IDF feature."""
    return "skill_" + re.sub(r"[^a-z0-9]+", "_", skill.lower()).strip("_")


def extract_skills(text: str) -> List[str]:
    """Skills literally present in the text (facts), in order of first appearance."""
    found: List[str] = []
    for m in _SKILL_RX.finditer(text or ""):
        canon = _ALIAS_TO_CANON[m.group(1).lower()]
        if canon not in found:
            found.append(canon)
    return found


def infer_skills(extracted: List[str]) -> List[str]:
    """Skills implied by extracted ones but not themselves stated in the text."""
    inferred: List[str] = []
    for s in extracted:
        for implied in IMPLIED_SKILLS.get(s, []):
            if implied not in extracted and implied not in inferred:
                inferred.append(implied)
    return inferred


def normalize_skill_tokens(text: str) -> str:
    """Replace each skill mention with one canonical token (e.g. 'scikit learn' -> skill_scikit_learn)."""
    return _SKILL_RX.sub(
        lambda m: " " + canonical_token(_ALIAS_TO_CANON[m.group(1).lower()]) + " ",
        text or "",
    )


@lru_cache(maxsize=None)
def _skill_rx(skill: str):
    aliases = [skill] + SKILL_TAXONOMY.get(skill, [])
    body = "|".join(re.escape(a.lower()) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![\w+#])({body})(?![\w+#])", re.IGNORECASE)


def mentions_skill(line: str, skill: str) -> bool:
    return bool(_skill_rx(skill).search(line or ""))


def find_evidence(text: str, skill: str, max_len: int = 150) -> Optional[str]:
    """A line of the resume that mentions the skill (for evidence-based explanations)."""
    rx = _skill_rx(skill)
    fallback = None
    for line in (text or "").splitlines():
        line = line.strip(" -•*\t")
        if rx.search(line):
            if len(line.split()) >= 6:  # prefer descriptive bullets over bare skill lists
                return line[:max_len] + ("..." if len(line) > max_len else "")
            fallback = fallback or line
    return fallback[:max_len] if fallback else None
