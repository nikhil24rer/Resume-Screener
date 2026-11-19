"""Matching logic: TF-IDF + cosine similarity, blended with skill coverage and experience fit.

Final score (0-100) = weighted blend of three *separately reported* components:
  1. text_similarity  - cosine similarity between TF-IDF vectors of JD and resume
  2. skill_match      - weighted share of JD skills found literally in the resume
  3. experience_fit   - candidate years / years requested in the JD (capped at 1)

It measures textual/keyword overlap with the JD only.  It does NOT measure
ability, potential, culture fit, or quality of work, and it must not replace
human judgement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import bullet_lines, explicit_years
from .preprocess import clean_for_tfidf
from .skills import extract_skills, find_evidence, mentions_skill

PREFERRED_RX = re.compile(r"nice[\s-]to[\s-]have|preferred|good\s+to\s+have|bonus|an?\s+advantage|plus\b", re.I)
REQUIRED_RX = re.compile(r"requirements?|responsibilit|qualifications?|must[\s-]have|required|what\s+you", re.I)


@dataclass
class Weights:
    text: float = 0.40
    skills: float = 0.45
    experience: float = 0.15

    def normalized(self, use_skills: bool = True, use_experience: bool = True) -> "Weights":
        w = Weights(self.text, self.skills if use_skills else 0.0, self.experience if use_experience else 0.0)
        total = w.text + w.skills + w.experience
        if total <= 0:
            return Weights(1.0, 0.0, 0.0)
        return Weights(w.text / total, w.skills / total, w.experience / total)


def parse_job_description(jd_text: str) -> Dict[str, object]:
    """Extract required / preferred skills and minimum years from a JD."""
    required, preferred, mode = [], [], "required"
    for line in (jd_text or "").splitlines():
        if PREFERRED_RX.search(line) and len(line.split()) <= 8:   # a short heading line
            mode = "preferred"
        elif REQUIRED_RX.search(line) and len(line.split()) <= 6:
            mode = "required"
        for s in extract_skills(line):
            if mode == "required" and s not in required:
                required.append(s)
            elif mode == "preferred" and s not in preferred and s not in required:
                preferred.append(s)
    preferred = [s for s in preferred if s not in required]
    return {"required_skills": required, "preferred_skills": preferred,
            "min_years": explicit_years(jd_text), "text": jd_text}


def compute_text_similarity(jd_text: str, resume_texts: List[str]) -> np.ndarray:
    """TF-IDF (unigrams+bigrams, sublinear tf) fitted on JD + all resumes; cosine vs the JD."""
    docs = [clean_for_tfidf(jd_text)] + [clean_for_tfidf(t) for t in resume_texts]
    if not any(docs[0].split()):
        return np.zeros(len(resume_texts))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return np.zeros(len(resume_texts))
    return cosine_similarity(matrix[0], matrix[1:]).ravel()


def _relevant_experience(rec: Dict[str, object], jd_skills: List[str], limit: int = 3) -> List[str]:
    lines = bullet_lines(str(rec.get("experience_text", ""))) + bullet_lines(str(rec.get("projects_text", "")))
    scored = []
    for ln in lines:
        hits = sum(mentions_skill(ln, s) for s in jd_skills)
        if hits:
            scored.append((hits, ln))
    scored.sort(key=lambda t: -t[0])
    return [ln[:200] for _, ln in scored[:limit]]


def explain(rec: Dict[str, object], matched: List[str], missing: List[str], inferred_cover: List[str],
            total_jd_skills: int, min_years: Optional[float]) -> str:
    """Short, evidence-based explanation built only from facts found in the resume."""
    parts = []
    if total_jd_skills:
        parts.append(f"Confirmed {len(matched)}/{total_jd_skills} JD skills"
                     + (f" ({', '.join(matched[:6])}{'...' if len(matched) > 6 else ''})." if matched else "."))
    for s in matched[:2]:
        ev = find_evidence(str(rec["raw_text"]), s)
        if ev:
            parts.append(f'Evidence for {s}: "{ev}"')
    years = rec.get("experience_years") or 0
    if min_years:
        verdict = "meets" if years >= min_years else "is below"
        parts.append(f"{years:.1f} yrs experience {verdict} the {min_years:g}+ requested.")
    else:
        parts.append(f"{years:.1f} yrs experience.")
    if inferred_cover:
        parts.append(f"Possibly covers (inferred, not stated): {', '.join(inferred_cover)}.")
    if missing:
        parts.append(f"Not found in resume: {', '.join(missing)}.")
    return " ".join(parts)


def rank_candidates(records: List[Dict[str, object]], jd_text: str,
                    weights: Optional[Weights] = None) -> pd.DataFrame:
    """Score and rank all candidates against the JD. Returns a DataFrame sorted by match_score."""
    weights = weights or Weights()
    jd = parse_job_description(jd_text)
    required, preferred = jd["required_skills"], jd["preferred_skills"]
    all_jd_skills = required + preferred
    min_years = jd["min_years"]

    w = weights.normalized(use_skills=bool(all_jd_skills), use_experience=bool(min_years))
    text_sims = compute_text_similarity(jd_text, [str(r["raw_text"]) for r in records]) if records else []

    rows = []
    for rec, tsim in zip(records, text_sims):
        have = set(rec["skills"])
        matched = [s for s in all_jd_skills if s in have]
        missing = [s for s in all_jd_skills if s not in have]
        inferred_cover = [s for s in missing if s in set(rec.get("inferred_skills", []))]
        truly_missing = [s for s in missing if s not in inferred_cover]

        denom = len(required) * 1.0 + len(preferred) * 0.5
        gained = sum(1.0 for s in matched if s in required) + sum(0.5 for s in matched if s in preferred)
        skill_score = gained / denom if denom else 0.0

        years = float(rec.get("experience_years") or 0)
        exp_score = min(years / min_years, 1.0) if min_years else 0.0

        final = w.text * tsim + w.skills * skill_score + w.experience * exp_score
        rows.append({
            "name": rec["name"],
            "match_score": round(100 * final, 1),
            "text_similarity": round(100 * float(tsim), 1),
            "skill_match": round(100 * skill_score, 1),
            "experience_fit": round(100 * exp_score, 1) if min_years else None,
            "experience_years": round(years, 1),
            "education_level": rec.get("education_level", "Not found"),
            "extracted_skills": rec["skills"],
            "matched_skills": matched,
            "missing_skills": truly_missing,
            "inferred_skills": inferred_cover,
            "relevant_experience": _relevant_experience(rec, all_jd_skills),
            "explanation": explain(rec, matched, truly_missing, inferred_cover, len(all_jd_skills), min_years),
            "email": rec.get("email"),
            "source": rec.get("source", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["match_score", "skill_match"], ascending=False, kind="mergesort").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.attrs["weights_used"] = w
    df.attrs["jd"] = jd
    return df


def report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flat, export-friendly version (lists -> comma-separated strings)."""
    out = df.copy()
    for col in ("extracted_skills", "matched_skills", "missing_skills", "inferred_skills"):
        out[col] = out[col].apply(lambda xs: ", ".join(xs))
    out["relevant_experience"] = out["relevant_experience"].apply(lambda xs: " | ".join(xs))
    return out
