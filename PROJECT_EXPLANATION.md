# Project Explanation

## 1. Workflow
```
JD text ─────────────────────────────┐
PDF resumes → pdfplumber/PyMuPDF → text ─┤→ parse (name, sections, skills, education, years)
Synthetic CSV → clean (dupes, NaN) ──────┘            │
                                                       ▼
                    clean_for_tfidf → TF-IDF (1-2 grams) → cosine(JD, resume)
                    skills ∩ JD skills  → skill coverage
                    years / required   → experience fit
                                                       ▼
                         weighted score → rank → top 5 + explanation
```

## 2. Data preparation (Day 1)
- 30 synthetic candidates across 5 roles; **injected problems**: 1 exact duplicate, 1 case/whitespace duplicate,
  2 missing experience values, 1 missing e-mail, 1 missing skills field, 1 empty projects field, inconsistent
  skill separators and casing.
- Cleaning: trim, blank→NaN, standardise name/e-mail case, drop empty rows, dedupe (e-mail, then name+skills),
  coerce `experience_years` to numeric. Missing skills are recovered by extracting from the full resume text;
  missing years are recovered from date ranges. Every step is counted in a log (32 → 30 rows).
- Dataset **and** PDFs go through the *same* parser, so behaviour is consistent.

## 3. Algorithms
**TF-IDF** weights a term by how often it appears in a document (sublinear tf = 1+log tf) and how rare it is across
documents (idf), so distinctive words ("nlp", "tf-idf") matter more than common ones ("team"). I use unigrams+bigrams
and English stop-words. Skill mentions are first normalised to single tokens (`scikit learn`, `sklearn`, `Scikit-Learn`
→ `skill_scikit_learn`) so spelling variants map to one feature.

**Cosine similarity** = cos θ = (a·b)/(‖a‖‖b‖): the angle between the JD vector and the resume vector. Ignores
document length, range 0–1 for TF-IDF vectors.

**Why not only TF-IDF?** Absolute cosine values between a short JD and a long resume are small and hard to interpret
(≈0.00–0.16 on the sample data), and TF-IDF can't tell "required" from "nice-to-have". So the final score also uses transparent
skill coverage and experience fit.

**Skill coverage** = (required matches ×1 + preferred matches ×0.5) / (required ×1 + preferred ×0.5).
**Experience fit** = min(years / required years, 1).

**Facts vs inference:** extracted skills appear literally in the text. Inferred ones (pandas ⇒ Python) are listed as
"possibly covered", never counted as matches.

## 4. What the score means / does not mean
It means: *how much this resume's wording, skills and years overlap with this JD.* It does **not** measure true
ability, depth, quality, potential, soft skills, or skills outside the taxonomy. No sensitive attributes are extracted
or used. A high score is a reason to read the resume, not to hire.

## 5. Testing
- 18 unit tests: skill extraction edge cases (c++, c#, java vs javascript, mysql vs sql), date-range merging, JD
  parsing, cleaning, invalid/scanned PDFs, ranking order, bounded scores, empty inputs.
- `evaluate.py`: for 4 JDs, precision@5 against the role each synthetic candidate was generated for.
  Result: 1.00 for all weightings. **Interpretation:** the pipeline works end-to-end, but this is *not* evidence of
  real-world accuracy – synthetic resumes come from templates that share role vocabulary, so any method wins.
- Bugs found by tests: an off-by-one in date-range months (Dec treated as exclusive); an ambiguous
  conditional for the experience scope.
- Edge cases to mention: a resume that merely lists many skills can score well without depth; candidates using
  different words for the same idea are penalised on the TF-IDF part; keyword stuffing would inflate scores.

## 6. Limitations & improvements (next week)
No OCR (add Tesseract); semantic matching with sentence embeddings (e.g. SBERT) alongside TF-IDF; larger/learned
skill ontology (e.g. ESCO) or NER model; section-aware weighting (skills in a project > skills in a list);
evaluation on a small human-labelled set; bias audit; recency weighting of experience; user feedback loop to tune weights.

## 7. Interview Q&A (read, understand, then answer in your own words)
1. **Why TF-IDF?** Fast, interpretable, no training data needed, works with tiny corpora, good baseline for keyword-heavy
   documents like resumes. Weak at synonyms/semantics – embeddings would be the upgrade.
2. **AI vs ML vs DL?** AI = machines doing tasks that need intelligence (includes rule-based systems); ML = AI that learns
   patterns from data; DL = ML using multi-layer neural networks. Note: *this project is mostly classical NLP + rules;
   it has no trained model* – say so.
3. **How does cosine similarity work here?** See §3.
4. **Missing/duplicate/inconsistent data?** See §2.
5. **How are skills extracted?** Taxonomy + alias map + regex with custom word boundaries, longest-match first.
6. **Resume skills not in the JD?** Ignored for skill coverage (they can't lower or raise it); they only slightly affect
   TF-IDF through the vector norm. Tested in `test_unrelated_skills_do_not_help`.
7. **How is the score calculated / what doesn't it measure?** §3–4.
8. **Testing / unexpected results?** §5.
9. **Limitations?** §6.
10. **Another week?** §6.
11. **Walk through a function:** be ready to walk through `rank_candidates` in `src/matcher.py` and
    `experience_years_from_dates` in `src/parser.py` line by line. **Do this before the interview.**
