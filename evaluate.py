"""Sanity-check the ranking against the synthetic role labels.

For each test JD we ask: of the top-5 candidates, how many were *generated* for the
matching role?  `role_label` is only used here, never as a ranking feature.

Caveat (be honest in your write-up): the synthetic resumes are template-generated,
so high precision here shows the pipeline works, NOT that it would work on messy
real-world resumes.

Run:  python evaluate.py
"""
import pandas as pd

from src.matcher import Weights, rank_candidates
from src.pipeline import records_from_dataframe

RAW = "data/sample_candidates_raw.csv"

TEST_JDS = {
    "data_scientist": open("data/sample_job_description.txt", encoding="utf-8").read(),
    "web_developer": """Frontend Developer
Requirements
- 2+ years of experience building web apps
- JavaScript, React, HTML and CSS
- Node.js and REST API development
- Git and agile workflows
Nice to have
- TypeScript and MongoDB
""",
    "devops_engineer": """DevOps Engineer
Requirements
- 3+ years of experience in cloud infrastructure
- Docker and Kubernetes
- AWS, Terraform and Linux
- CI/CD with Jenkins
Nice to have
- Bash scripting and Azure
""",
    "data_analyst": """Data Analyst
Requirements
- 1+ years of experience in analytics
- Strong SQL and Excel
- Power BI or Tableau dashboards
- Data analysis and data visualization
Nice to have
- Python and pandas
""",
}


def main(k: int = 5) -> None:
    raw = pd.read_csv(RAW)
    records = records_from_dataframe(raw)
    labels = raw.drop_duplicates("email").set_index("name")["role_label"].to_dict()
    labels.update(raw.assign(name=raw["name"].str.strip().str.title()).set_index("name")["role_label"].to_dict())

    configs = {"default (0.40/0.45/0.15)": Weights(),
               "tf-idf only": Weights(1, 0, 0),
               "skills only": Weights(0, 1, 0)}
    print(f"{'JD':<18}{'weights':<26}{'precision@5':>12}")
    for jd_name, jd in TEST_JDS.items():
        for cfg_name, w in configs.items():
            ranked = rank_candidates(records, jd, w)
            top = ranked.head(k)["name"].map(labels)
            p = (top == jd_name).mean()
            print(f"{jd_name:<18}{cfg_name:<26}{p:>12.2f}")


if __name__ == "__main__":
    main()
