"""Run with:  python -m unittest discover -s tests -v   (or `pytest` if installed)."""
import io
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matcher import Weights, parse_job_description, rank_candidates  # noqa: E402
from src.parser import (education_level, experience_years_from_dates, explicit_years,  # noqa: E402
                        extract_name, parse_resume_text)
from src.pdf_parser import ResumeParseError, extract_text_from_pdf  # noqa: E402
from src.pipeline import records_from_dataframe  # noqa: E402
from src.preprocess import clean_candidate_dataframe  # noqa: E402
from src.skills import extract_skills, infer_skills  # noqa: E402


class SkillTests(unittest.TestCase):
    def test_aliases_and_symbols(self):
        s = extract_skills("Used sklearn, Node.js, C++ and CI/CD with PostgreSQL")
        for expected in ("scikit-learn", "node.js", "c++", "ci/cd", "postgresql"):
            self.assertIn(expected, s)

    def test_java_is_not_javascript(self):
        self.assertEqual(extract_skills("Wrote JavaScript daily"), ["javascript"])
        self.assertEqual(extract_skills("Wrote Java daily"), ["java"])

    def test_no_substring_false_positives(self):
        self.assertNotIn("sql", extract_skills("MySQL"))           # mysql is its own skill
        self.assertNotIn("r", extract_skills("I prefer working in teams"))

    def test_inferred_not_counted_as_extracted(self):
        ex = extract_skills("Built models with pandas")
        self.assertNotIn("python", ex)
        self.assertIn("python", infer_skills(ex))


class ParserTests(unittest.TestCase):
    def test_name(self):
        self.assertEqual(extract_name("Asha Verma\nasha@x.com\nSKILLS\npython"), "Asha Verma")
        self.assertEqual(extract_name("SKILLS\npython", "Fallback"), "Fallback")

    def test_date_ranges_merge_overlap(self):
        txt = "Jan 2020 - Dec 2020\nJun 2020 - Dec 2021"   # overlapping -> Jan 2020..Dec 2021 = 24 months
        self.assertAlmostEqual(experience_years_from_dates(txt, today=date(2026, 1, 1)), 2.0, places=1)

    def test_present_and_explicit(self):
        self.assertGreater(experience_years_from_dates("Jan 2024 - Present", today=date(2026, 1, 1)), 1.9)
        self.assertEqual(explicit_years("Minimum 3+ years of experience in ML"), 3.0)

    def test_education_level(self):
        self.assertEqual(education_level("M.Tech in CS")["label"], "Master's")
        self.assertEqual(education_level("B.Sc Statistics")["label"], "Bachelor's")

    def test_education_dates_not_counted_as_work(self):
        text = "Jo Doe\nEDUCATION\nB.Tech 2016 - 2020\nEXPERIENCE\nAnalyst | Acme | Jan 2022 - Dec 2023\n"
        self.assertAlmostEqual(parse_resume_text(text)["experience_years"], 2.0, places=0)


class CleaningTests(unittest.TestCase):
    def test_duplicates_missing_and_casing(self):
        df = pd.DataFrame({
            "name": ["ann lee", "ANN LEE ", None, "Bob Ray"],
            "email": ["ann@x.com", "ANN@X.COM", None, ""],
            "skills": ["python", "python", None, "sql"],
            "experience_years": ["3", "3", None, "abc"],
        })
        clean, log = clean_candidate_dataframe(df)
        self.assertEqual(len(clean), 2)                      # ann duplicate + empty row removed
        self.assertEqual(clean.loc[0, "name"], "Ann Lee")
        self.assertTrue(pd.isna(clean.loc[1, "experience_years"]))   # 'abc' -> NaN, not a crash
        self.assertEqual(log["dropped_empty_rows"], 1)


class PdfTests(unittest.TestCase):
    def test_rejects_non_pdf_empty_and_corrupt(self):
        for bad in (b"", b"plain text", b"%PDF-1.4\nnot really"):
            with self.assertRaises(ResumeParseError):
                extract_text_from_pdf(bad)

    def test_scanned_pdf_detected(self):
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.rect(10, 10, 100, 100, fill=1)
        c.save()
        with self.assertRaises(ResumeParseError) as ctx:
            extract_text_from_pdf(buf.getvalue())
        self.assertIn("scanned", str(ctx.exception).lower())


class MatcherTests(unittest.TestCase):
    JD = "Data Scientist\nRequirements\n- 2+ years of experience\n- Python, pandas, scikit-learn, NLP\nNice to have\n- Docker\n"

    def _records(self):
        df = pd.DataFrame([
            {"name": "Strong Match", "email": "a@x.com", "skills": "python, pandas, scikit-learn, nlp, docker",
             "experience": "Data Scientist | X | Jan 2021 - Present\n- Built nlp models with python and scikit-learn for ticket routing",
             "experience_years": 4},
            {"name": "Weak Match", "email": "b@x.com", "skills": "html, css, javascript",
             "experience": "Web Dev | Y | Jan 2025 - Present\n- Built landing pages with html and css for clients",
             "experience_years": 1},
        ])
        return records_from_dataframe(df)

    def test_jd_parsing(self):
        jd = parse_job_description(self.JD)
        self.assertEqual(set(jd["required_skills"]), {"python", "pandas", "scikit-learn", "nlp"})
        self.assertEqual(jd["preferred_skills"], ["docker"])
        self.assertEqual(jd["min_years"], 2.0)

    def test_ranking_order_and_missing_skills(self):
        df = rank_candidates(self._records(), self.JD)
        self.assertEqual(df.loc[0, "name"], "Strong Match")
        self.assertGreater(df.loc[0, "match_score"], df.loc[1, "match_score"])
        self.assertIn("python", df.loc[1, "missing_skills"])
        self.assertEqual(df.loc[0, "missing_skills"], [])

    def test_scores_bounded(self):
        df = rank_candidates(self._records(), self.JD)
        self.assertTrue(df["match_score"].between(0, 100).all())

    def test_weights_change_result_and_normalise(self):
        w = Weights(2, 2, 1).normalized()
        self.assertAlmostEqual(w.text + w.skills + w.experience, 1.0)

    def test_empty_and_skillless_jd_do_not_crash(self):
        self.assertTrue(rank_candidates([], self.JD).empty)
        df = rank_candidates(self._records(), "We are hiring a thoughtful colleague.")
        self.assertEqual(len(df), 2)

    def test_unrelated_skills_do_not_help(self):
        # a resume with extra skills not in the JD must not outrank a resume covering the JD
        recs = self._records()
        recs[1]["skills"] = recs[1]["skills"] + ["kubernetes", "terraform", "aws", "linux"]
        df = rank_candidates(recs, self.JD)
        self.assertEqual(df.loc[0, "name"], "Strong Match")


if __name__ == "__main__":
    unittest.main()
