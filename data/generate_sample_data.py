"""Generate fully synthetic candidates (no real people), messy on purpose.

Outputs (in ./data):
  sample_candidates_raw.csv        30 candidates + injected duplicates / missing values
  sample_candidates_processed.csv  after cleaning + parsing (what the matcher sees)
  sample_resumes/*.pdf             a few candidates rendered as PDF resumes
  sample_job_description.txt       demo JD (Data Scientist, NLP)

`role_label` is used ONLY by evaluate.py to sanity-check rankings; it is never a ranking feature.
Run from the project root:  python data/generate_sample_data.py
"""
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import compose_resume_text, records_from_dataframe  # noqa: E402

random.seed(42)
DATA = ROOT / "data"

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vikram", "Ananya", "Arjun", "Sara",
         "Neha", "Rahul", "Priya", "Karan", "Tanvi", "Aditya", "Sneha", "Manav", "Riya", "Yash",
         "Pooja", "Nikhil", "Kavya", "Sameer", "Divya", "Harsh", "Ira", "Dev", "Lata", "Omar"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Gupta", "Reddy", "Mehta", "Kapoor", "Singh", "Das",
        "Joshi", "Patel", "Bose", "Menon", "Rao", "Khanna", "Jain", "Chopra", "Malik", "Saxena"]
COMPANIES = ["Nimbus Labs", "Quantix Systems", "BrightPath Tech", "Orbit Analytics", "Kestrel Software",
             "Lumen Data", "Vertex Cloud", "Pixel Forge", "Atlas Digital", "Helio Works"]
UNIS = ["Northfield Institute of Technology", "Riverside University", "Summit College of Engineering",
        "Lakeshore University", "Crestview Institute of Science"]
DEGREES = [("B.Tech in Computer Science", 2), ("B.Sc in Statistics", 2), ("B.E in Information Technology", 2),
           ("M.Tech in Computer Science", 3), ("M.Sc in Data Science", 3), ("MCA", 3), ("BCA", 2)]

ROLES = {
    "data_scientist": dict(
        n=8, titles=["Data Scientist", "Junior Data Scientist", "Machine Learning Analyst"],
        pool=["python", "pandas", "numpy", "scikit-learn", "machine learning", "statistics", "sql",
              "tf-idf", "nlp", "feature engineering", "xgboost", "matplotlib"],
        bullets=["Built {a} models with {b} and {c} to predict customer churn, improving recall by {n}%",
                 "Applied {a} and {b} for text classification of support tickets using {c}",
                 "Performed feature engineering with {a} and {b}; evaluated models with cross validation using {c}"],
        projects=["Resume matching tool: used {a} and {b} to rank documents by similarity",
                  "Churn prediction: trained {a} model with {b} and tracked metrics with {c}"]),
    "ml_engineer": dict(
        n=6, titles=["ML Engineer", "Deep Learning Engineer", "AI Engineer"],
        pool=["python", "pytorch", "tensorflow", "deep learning", "docker", "mlops", "aws", "fastapi",
              "nlp", "transformers", "git", "model deployment"],
        bullets=["Trained {a} models in {b} and deployed them with {c}, cutting inference latency by {n}%",
                 "Built {a} pipelines and containerised services with {b} on {c}",
                 "Fine-tuned {a} models for NLP tasks using {b} and served them via {c}"],
        projects=["Sentiment API: fine-tuned {a} with {b} and exposed it using {c}",
                  "Image classifier: built with {a} and {b}, packaged using {c}"]),
    "web_developer": dict(
        n=6, titles=["Web Developer", "Frontend Developer", "Full Stack Developer"],
        pool=["javascript", "react", "node.js", "html", "css", "mongodb", "rest api", "git",
              "typescript", "mysql", "agile"],
        bullets=["Developed responsive interfaces with {a} and {b}, improving page load time by {n}%",
                 "Designed {a} backends using {b} and {c}",
                 "Implemented features in {a} and {b} following agile practices with {c}"],
        projects=["E-commerce site built with {a}, {b} and {c}",
                  "Task manager app using {a} and {b} with a {c} backend"]),
    "data_analyst": dict(
        n=5, titles=["Data Analyst", "Business Analyst", "BI Analyst"],
        pool=["sql", "excel", "power bi", "tableau", "data analysis", "python", "pandas",
              "data visualization", "statistics", "etl"],
        bullets=["Created dashboards in {a} and {b} used by leadership, saving {n} hours weekly",
                 "Wrote complex {a} queries and automated reports with {b} and {c}",
                 "Performed data analysis with {a} and {b} to uncover revenue trends"],
        projects=["Sales dashboard built with {a} and {b}",
                  "Customer segmentation using {a}, {b} and {c}"]),
    "devops_engineer": dict(
        n=5, titles=["DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer"],
        pool=["docker", "kubernetes", "aws", "terraform", "linux", "jenkins", "ci/cd", "git",
              "bash", "azure"],
        bullets=["Automated deployments using {a} and {b}, reducing release time by {n}%",
                 "Managed infrastructure as code with {a} on {b} and {c}",
                 "Built CI/CD pipelines in {a} with {b} and {c}"],
        projects=["Kubernetes cluster automation with {a} and {b}",
                  "Infrastructure provisioning using {a}, {b} and {c}"]),
}
GENERIC = ["git", "agile", "unit testing", "linux"]
SEPARATORS = [", ", "; ", " | ", ",", " / "]
CASINGS = [str, str.lower, str.upper, str.title]


def month_year(offset_months: int) -> str:
    today = date.today()
    idx = today.year * 12 + today.month - 1 - offset_months
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][idx % 12]} {idx // 12}"


def make_experience(role: dict, skills: list, years: float) -> str:
    if years <= 0:
        return ""
    total_months = int(years * 12)
    n_jobs = 2 if total_months >= 30 else 1
    cuts = [total_months] if n_jobs == 1 else [total_months // 2, total_months - total_months // 2]
    lines, offset_end = [], 0
    for i, dur in enumerate(cuts):
        start = offset_end + dur
        end_txt = "Present" if offset_end == 0 else month_year(offset_end)
        lines.append(f"{random.choice(role['titles'])} | {random.choice(COMPANIES)} | {month_year(start)} - {end_txt}")
        for tmpl in random.sample(role["bullets"], 2):
            a, b, c = (random.sample(skills, 3) if len(skills) >= 3 else (skills * 3)[:3])
            lines.append("- " + tmpl.format(a=a, b=b, c=c, n=random.randint(8, 40)))
        offset_end = start
    return "\n".join(lines)


def make_projects(role: dict, skills: list) -> str:
    lines = []
    for tmpl in random.sample(role["projects"], 2):
        a, b, c = (random.sample(skills, 3) if len(skills) >= 3 else (skills * 3)[:3])
        lines.append("- " + tmpl.format(a=a, b=b, c=c))
    return "\n".join(lines)


def build_candidates() -> pd.DataFrame:
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    random.shuffle(names)
    rows, i = [], 0
    for role_key, role in ROLES.items():
        for _ in range(role["n"]):
            name = names[i]
            i += 1
            core = random.sample(role["pool"], k=random.randint(6, min(9, len(role["pool"]))))
            extras = random.sample(GENERIC, k=random.randint(0, 2))
            skills = list(dict.fromkeys(core + extras))
            years = random.choice([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
            deg = random.choice(DEGREES)
            grad = date.today().year - int(years) - 1
            sep, cas = random.choice(SEPARATORS), random.choice(CASINGS)
            rows.append({
                "candidate_id": f"C{i:03d}",
                "name": name,
                "email": f"{name.lower().replace(' ', '.')}@example.com",
                "skills": cas(sep.join(skills)),
                "education": f"{deg[0]}, {random.choice(UNIS)} ({grad - 3} - {grad})",
                "experience_years": years,
                "experience": make_experience(role, skills, years),
                "projects": make_projects(role, skills),
                "role_label": role_key,
            })
    df = pd.DataFrame(rows)

    # ---- inject realistic data problems for the cleaning step to handle ----
    dup_exact = df.iloc[[2]].copy()                                   # exact duplicate
    dup_case = df.iloc[[9]].copy()                                    # same person, messy name/email
    dup_case["name"] = dup_case["name"].str.upper() + "  "
    dup_case["email"] = dup_case["email"].str.upper()
    dup_case["candidate_id"] = "C901"
    df = pd.concat([df, dup_exact, dup_case], ignore_index=True)
    df.loc[5, "experience_years"] = None                              # missing experience (recoverable from dates)
    df.loc[14, "experience_years"] = None
    df.loc[7, "email"] = None                                         # missing e-mail
    df.loc[11, "skills"] = None                                       # missing skills column (recovered from text)
    df.loc[20, "projects"] = ""                                       # empty projects
    return df


def write_pdfs(processed_raw: pd.DataFrame, picks: list) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    head = ParagraphStyle("h", parent=styles["Heading2"], spaceBefore=8, spaceAfter=2)
    body = styles["BodyText"]
    out_dir = DATA / "sample_resumes"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.pdf"):
        old.unlink()
    for idx in picks:
        row = processed_raw.iloc[idx].to_dict()
        text = compose_resume_text(row)
        story = []
        for line in text.splitlines():
            if not line.strip():
                story.append(Spacer(1, 4))
            elif line in ("SKILLS", "EDUCATION", "EXPERIENCE", "PROJECTS"):
                story.append(Paragraph(line, head))
            else:
                story.append(Paragraph(escape(line), body))
        fname = row["name"].replace(" ", "_") + "_Resume.pdf"
        SimpleDocTemplate(str(out_dir / fname), pagesize=A4).build(story)


JD = """Data Scientist (NLP)

About the role
We are looking for a Data Scientist to build text-matching and classification models that help our HR team screen applications faster.

Requirements
- 2+ years of experience in data science or machine learning
- Strong Python with pandas and numpy
- Hands-on experience with scikit-learn and machine learning model development
- Experience with NLP and TF-IDF based text representation
- Solid understanding of statistics and SQL
- Familiarity with git

Nice to have
- Experience with deep learning (PyTorch or TensorFlow)
- Docker and model deployment
- Feature engineering and XGBoost
"""

if __name__ == "__main__":
    raw = build_candidates()
    raw.to_csv(DATA / "sample_candidates_raw.csv", index=False)

    records = records_from_dataframe(raw)
    processed = pd.DataFrame([{
        "name": r["name"], "email": r["email"], "skills": ", ".join(r["skills"]),
        "education": r["education"], "education_level": r["education_level"],
        "experience_years": r["experience_years"], "experience_source": r["experience_source"],
        "projects": r["projects_text"].replace("\n", " | "), "source": r["source"],
    } for r in records])
    processed.to_csv(DATA / "sample_candidates_processed.csv", index=False)

    picks = [0, 1, 8, 9, 15, 16, 21, 22, 26, 27]   # a mix of roles
    write_pdfs(raw, picks)
    (DATA / "sample_job_description.txt").write_text(JD, encoding="utf-8")
    print(f"raw rows: {len(raw)}  ->  processed rows: {len(processed)}")
    print(f"PDF resumes written: {len(picks)}")
