"""Turn raw resume text into a structured record.

Everything here is rule-based (regex + section headings) so that each extracted
field can be explained.  Protected attributes (age, gender, photo, marital
status, etc.) are deliberately NOT extracted and never used in ranking.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional

from .skills import extract_skills, infer_skills

SECTION_PATTERNS = {
    "summary": r"(professional\s+)?(summary|profile|objective|about\s+me)",
    "skills": r"(technical\s+|key\s+|core\s+)?(skills|competencies|technologies|tech\s+stack)",
    "education": r"(education|academic\s+background|qualifications?)",
    "experience": r"(work\s+|professional\s+|relevant\s+)?(experience|employment(\s+history)?|work\s+history)",
    "projects": r"(personal\s+|academic\s+|key\s+)?projects?",
}
_SECTION_RX = {k: re.compile(rf"^\s*{v}\s*:?\s*$", re.IGNORECASE) for k, v in SECTION_PATTERNS.items()}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

DATE_RANGE_RX = re.compile(
    r"(?P<sm>[A-Za-z]{3,9})?\.?\s*(?P<sy>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    r"(?:(?P<em>[A-Za-z]{3,9})?\.?\s*(?P<ey>(?:19|20)\d{2})|(?P<pres>present|current|now|till\s+date|ongoing))",
    re.IGNORECASE,
)
EXPLICIT_YEARS_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)[^.\n]{0,40}?experience"
    r"|experience[^.\n]{0,30}?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

EDU_LEVELS = [  # (rank, label, regex) - highest rank found wins
    (4, "PhD", re.compile(r"\b(ph\.?d|doctorate)\b", re.I)),
    (3, "Master's", re.compile(r"\b(master'?s?|m\.?\s?tech|m\.?\s?sc|mba|mca|m\.?\s?s)\b", re.I)),
    (2, "Bachelor's", re.compile(r"\b(bachelor'?s?|b\.?\s?tech|b\.?\s?e|b\.?\s?sc|bca|b\.?\s?s)\b", re.I)),
    (1, "Diploma", re.compile(r"\b(diploma|associate)\b", re.I)),
]


def split_sections(text: str) -> Dict[str, str]:
    """Split resume text into sections using heading lines; unmatched leading text -> 'header'."""
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 40:
            heading = next((k for k, rx in _SECTION_RX.items() if rx.match(stripped)), None)
            if heading:
                current = heading
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_name(text: str, fallback: str = "Unknown") -> str:
    """First of the top lines that looks like a person's name (2-4 alphabetic words)."""
    for line in (text or "").splitlines()[:6]:
        line = line.strip()
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", w) for w in words):
            if not any(rx.match(line) for rx in _SECTION_RX.values()):
                return " ".join(words).title()
    return fallback


def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RX.search(text or "")
    return m.group(0).lower() if m else None


def _month_index(year: int, month_word: Optional[str], default: int) -> int:
    month = MONTHS.get((month_word or "").lower()[:3], default)
    return year * 12 + month


def experience_years_from_dates(text: str, today: Optional[date] = None) -> float:
    """Sum of (merged) date ranges such as 'Jan 2021 - Present' -> years."""
    today = today or date.today()
    now_idx = today.year * 12 + today.month
    spans = []
    for m in DATE_RANGE_RX.finditer(text or ""):
        start = _month_index(int(m.group("sy")), m.group("sm"), 1)
        if m.group("pres"):
            end = now_idx
        else:  # explicit end month is inclusive ("... - Dec 2021" includes December)
            end = min(_month_index(int(m.group("ey")), m.group("em"), 12) + 1, now_idx + 1)
        if end > start:
            spans.append((start, end))
    spans.sort()
    total, cur_s, cur_e = 0, None, None
    for s, e in spans:  # merge overlapping intervals so parallel jobs aren't double counted
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return round(total / 12, 1)


def explicit_years(text: str) -> Optional[float]:
    """Years stated outright, e.g. '3+ years of experience'. Returns the largest mention."""
    vals = []
    for m in EXPLICIT_YEARS_RX.finditer(text or ""):
        vals.append(float(m.group(1) or m.group(2)))
    vals = [v for v in vals if v <= 45]
    return max(vals) if vals else None


def education_level(text: str) -> Dict[str, object]:
    best_rank, best_label = 0, "Not found"
    for rank, label, rx in EDU_LEVELS:
        if rx.search(text or "") and rank > best_rank:
            best_rank, best_label = rank, label
    return {"rank": best_rank, "label": best_label}


def bullet_lines(section_text: str) -> List[str]:
    lines = [ln.strip(" -•*\t") for ln in (section_text or "").splitlines()]
    return [ln for ln in lines if len(ln.split()) >= 4]


def parse_resume_text(text: str, fallback_name: str = "Unknown") -> Dict[str, object]:
    """Main entry: raw text -> structured candidate record."""
    sections = split_sections(text)
    exp_text = sections.get("experience", "")
    edu_text = sections.get("education", "")

    # date-derived experience from the experience section (or whole text minus education)
    if exp_text:
        scope = exp_text
    elif edu_text:
        scope = text.replace(edu_text, "")  # avoid counting graduation-year ranges as work
    else:
        scope = text
    from_dates = experience_years_from_dates(scope)
    stated = explicit_years(text)
    years = from_dates if from_dates > 0 else (stated or 0.0)

    skills = extract_skills(text)
    edu = education_level(edu_text or text)

    return {
        "name": extract_name(text, fallback_name),
        "email": extract_email(text),
        "skills": skills,
        "inferred_skills": infer_skills(skills),
        "education": edu_text.replace("\n", "; ") if edu_text else "Not found",
        "education_level": edu["label"],
        "education_rank": edu["rank"],
        "experience_years": years,
        "experience_text": exp_text,
        "projects_text": sections.get("projects", ""),
        "raw_text": text,
        "experience_source": "dates" if from_dates > 0 else ("stated" if stated else "none"),
    }
