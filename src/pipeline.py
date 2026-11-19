"""Glue code: dataset rows / uploaded PDFs -> structured candidate records."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .parser import parse_resume_text
from .pdf_parser import ResumeParseError, extract_text_from_pdf
from .preprocess import clean_candidate_dataframe


def compose_resume_text(row: Dict[str, object]) -> str:
    """Rebuild a resume-like text from structured dataset columns."""
    def val(key):
        v = row.get(key)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    parts = [val("name"), val("email"), ""]
    if val("skills"):
        parts += ["SKILLS", val("skills"), ""]
    if val("education"):
        parts += ["EDUCATION", val("education"), ""]
    if val("experience"):
        parts += ["EXPERIENCE", val("experience"), ""]
    if val("projects"):
        parts += ["PROJECTS", val("projects")]
    return "\n".join(parts).strip()


def records_from_dataframe(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Clean the table, then parse each row with the same parser used for PDFs."""
    clean_df, _ = clean_candidate_dataframe(df)
    records = []
    for _, row in clean_df.iterrows():
        rec = parse_resume_text(compose_resume_text(row.to_dict()), fallback_name=str(row.get("name") or "Unknown"))
        if pd.notna(row.get("name")):
            rec["name"] = row["name"]
        if pd.notna(row.get("email")):
            rec["email"] = row["email"]
        if pd.notna(row.get("experience_years")):       # trust a valid structured value
            rec["experience_years"] = float(row["experience_years"])
            rec["experience_source"] = "dataset column"
        rec["source"] = f"dataset:{row.get('candidate_id', '')}"
        records.append(rec)
    return records


def load_dataset(path: str | Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    raw = pd.read_csv(path)
    _, log = clean_candidate_dataframe(raw)
    return records_from_dataframe(raw), log


def records_from_pdfs(files: Iterable) -> Tuple[List[Dict[str, object]], List[Tuple[str, str]]]:
    """Parse uploaded PDFs. Returns (records, [(filename, reason_skipped), ...]); never raises per-file."""
    records, errors, seen_hashes, seen_emails = [], [], set(), set()
    for f in files:
        fname = getattr(f, "name", str(f))
        try:
            data = f.getvalue() if hasattr(f, "getvalue") else (f.read() if hasattr(f, "read") else Path(f).read_bytes())
            digest = hashlib.md5(data).hexdigest()
            if digest in seen_hashes:
                errors.append((fname, "Duplicate of an already uploaded file - skipped."))
                continue
            seen_hashes.add(digest)
            text = extract_text_from_pdf(data)
            rec = parse_resume_text(text, fallback_name=Path(fname).stem.replace("_", " ").title())
            if rec["email"] and rec["email"] in seen_emails:
                errors.append((fname, f"Same e-mail ({rec['email']}) as another upload - skipped as duplicate."))
                continue
            if rec["email"]:
                seen_emails.add(rec["email"])
            rec["source"] = fname
            records.append(rec)
        except ResumeParseError as exc:
            errors.append((fname, str(exc)))
        except Exception as exc:  # last-resort guard so one bad file never crashes the batch
            errors.append((fname, f"Unexpected error while reading file: {exc}"))
    return records, errors
