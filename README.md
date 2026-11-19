# Smart Resume Screening & Candidate Matching System

An explainable resume-screening prototype: upload a job description and multiple PDF resumes, get a ranked
shortlist with match scores, matched / missing skills, relevant-experience evidence and a short explanation.

> **AI-assistance disclosure:** _Fill this in honestly before submitting_ (e.g. "Initial code scaffold generated with
> Claude; I reviewed, tested and modified it. Changes I made: …").

## Features
- PDF text extraction with **pdfplumber** (+ optional **PyMuPDF** fallback); invalid, corrupt, empty, scanned and duplicate files are rejected with a clear reason instead of crashing
- Rule-based skill extraction (72 skills with aliases) – **extracted facts** and **inferred skills** are shown separately
- **TF-IDF + cosine similarity**, blended with skill coverage and experience fit (configurable weights)
- Top-5 ranking, per-candidate evidence-based explanation, missing skills
- Side-by-side comparison of two candidates, CSV / Excel report download
- 30 synthetic candidates (with injected duplicates & missing values) + cleaning pipeline

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Python 3.9+ recommended.

## Quick demo
1. Click **Load sample JD**.
2. Choose *Upload PDF resumes* and upload files from `data/sample_resumes/` (or choose *Sample dataset*).
3. Click **Screen candidates**.
4. Try the sidebar weights, the **Compare two** tab and the download buttons.
5. Upload a `.txt` renamed to `.pdf` to see graceful error handling.

## Project structure
```
app.py                       Streamlit UI
evaluate.py                  precision@5 sanity check on synthetic data
src/
  skills.py                  taxonomy, extraction, inference, evidence lookup
  pdf_parser.py              robust PDF text extraction + error handling
  parser.py                  name / sections / education / experience-years extraction
  preprocess.py              text cleaning + DataFrame cleaning (dupes, NaN, casing)
  matcher.py                 TF-IDF cosine, skill & experience scoring, explanations
  pipeline.py                dataset / PDFs -> structured records
data/
  generate_sample_data.py    builds the synthetic data & sample PDFs (seeded, reproducible)
  sample_candidates_raw.csv  32 messy rows -> 30 after cleaning
  sample_candidates_processed.csv
  sample_resumes/*.pdf       10 sample PDF resumes
  sample_job_description.txt
tests/test_core.py           18 unit tests
```

## Tests
```bash
python -m unittest discover -s tests -v
python evaluate.py
```

## Scoring (summary)
`match_score = 0.40·TF-IDF cosine + 0.45·skill coverage + 0.15·experience fit` (× 100; weights adjustable in the UI).
If the JD has no detectable skills or no years requirement, that component is dropped and the weights renormalised.
See `PROJECT_EXPLANATION.md` for details, limitations and interview Q&A.

## Limitations
- Scanned/image PDFs are detected but **not OCR'd**
- Skills outside the built-in taxonomy are invisible; no semantic understanding ("built recommender" ≠ "machine learning" unless listed)
- Multi-column / heavily designed PDFs may extract in jumbled order
- Experience years are derived from date ranges; unusual date formats may be missed
- Evaluated only on template-generated synthetic data
- A similarity score ≠ hiring suitability. Use to support, not replace, human review
