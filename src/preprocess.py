"""Text and tabular data cleaning (Day 1: cleaning, missing values, duplicates)."""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .skills import normalize_skill_tokens

EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RX = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RX = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")


def clean_for_tfidf(text: str) -> str:
    """Lowercase, drop contact details/URLs/numbers, unify skill spellings into single tokens."""
    text = str(text or "")
    text = EMAIL_RX.sub(" ", text)
    text = URL_RX.sub(" ", text)
    text = PHONE_RX.sub(" ", text)
    text = normalize_skill_tokens(text)          # 'Scikit Learn' / 'sklearn' -> skill_scikit_learn
    text = text.lower()
    text = re.sub(r"[^a-z_\s]", " ", text)       # drop digits & punctuation (keeps underscores)
    return re.sub(r"\s+", " ", text).strip()


def _blank_to_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: np.nan if (isinstance(v, str) and not v.strip()) else v)


def clean_candidate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean a raw candidate table.

    Steps (all counted in the returned log so they can be reported/explained):
      1. normalise column names, trim whitespace, blank -> NaN
      2. standardise name casing and lowercase e-mails
      3. drop rows that have neither a name nor any resume content
      4. remove duplicates (same e-mail, then same name + same skills)
      5. coerce experience_years to numeric; leave NaN for later text-based recovery
    """
    log: Dict[str, int] = {"rows_in": len(df)}
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = _blank_to_nan(df[col])

    if "name" in df:
        df["name"] = df["name"].apply(lambda v: " ".join(v.split()).title() if isinstance(v, str) else v)
    if "email" in df:
        df["email"] = df["email"].apply(lambda v: v.lower() if isinstance(v, str) else v)

    content_cols = [c for c in ("skills", "education", "experience", "projects") if c in df]
    empty_mask = df.get("name", pd.Series(np.nan, index=df.index)).isna() & df[content_cols].isna().all(axis=1)
    log["dropped_empty_rows"] = int(empty_mask.sum())
    df = df[~empty_mask]

    before = len(df)
    if "email" in df:
        has_email = df["email"].notna()
        dup_email = has_email & df.duplicated(subset="email", keep="first")
        df = df[~dup_email]
    log["duplicates_by_email"] = before - len(df)

    before = len(df)
    key_cols = [c for c in ("name", "skills") if c in df]
    if key_cols:
        norm = df[key_cols].astype(str).apply(lambda col: col.str.lower().str.replace(r"\W+", "", regex=True))
        df = df[~norm.duplicated(keep="first")]
    log["duplicates_by_name_skills"] = before - len(df)

    if "experience_years" in df:
        df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
        df.loc[df["experience_years"] < 0, "experience_years"] = np.nan
    else:
        df["experience_years"] = np.nan

    log["missing_name"] = int(df["name"].isna().sum()) if "name" in df else 0
    log["missing_email"] = int(df["email"].isna().sum()) if "email" in df else 0
    log["missing_skills_field"] = int(df["skills"].isna().sum()) if "skills" in df else 0
    log["missing_experience_years"] = int(df["experience_years"].isna().sum())
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log
