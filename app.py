"""Smart Resume Screening & Candidate Matching - Streamlit UI.

Run:  streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.matcher import Weights, rank_candidates, report_dataframe
from src.pdf_parser import ResumeParseError, extract_text_from_pdf
from src.pipeline import load_dataset, records_from_pdfs

ROOT = Path(__file__).parent
SAMPLE_CSV = ROOT / "data" / "sample_candidates_raw.csv"
SAMPLE_JD = ROOT / "data" / "sample_job_description.txt"

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def cached_sample_dataset():
    return load_dataset(SAMPLE_CSV)


def load_sample_jd():
    st.session_state["jd_text"] = SAMPLE_JD.read_text(encoding="utf-8")


def load_jd_from_file():
    f = st.session_state.get("jd_file")
    st.session_state["jd_file_error"] = None
    if f is None:
        return
    try:
        if f.name.lower().endswith(".pdf"):
            st.session_state["jd_text"] = extract_text_from_pdf(f.getvalue())
        else:
            st.session_state["jd_text"] = f.getvalue().decode("utf-8", errors="ignore")
    except ResumeParseError as exc:
        st.session_state["jd_file_error"] = f"Could not read job description file: {exc}"
    except Exception as exc:
        st.session_state["jd_file_error"] = f"Unexpected error reading file: {exc}"


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranking")
    return buf.getvalue()


def tag_list(items, icon):
    return "  ".join(f"`{icon} {i}`" for i in items) if items else "_none_"


def candidate_card(row):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match score", f"{row.match_score:.1f}")
    c2.metric("Text similarity (TF-IDF)", f"{row.text_similarity:.1f}")
    c3.metric("Skill match", f"{row.skill_match:.1f}")
    c4.metric("Experience", f"{row.experience_years:.1f} yrs")
    st.markdown(f"**Education level:** {row.education_level}")
    st.markdown(f"**Matched JD skills (found in resume):** {tag_list(row.matched_skills, '✅')}")
    st.markdown(f"**Missing / unmatched JD skills:** {tag_list(row.missing_skills, '❌')}")
    if row.inferred_skills:
        st.markdown(f"**Possibly covered (INFERRED, not stated):** {tag_list(row.inferred_skills, '⚠️')}")
    st.markdown(f"**All extracted skills:** {', '.join(row.extracted_skills) or '_none found_'}")
    if row.relevant_experience:
        st.markdown("**Relevant experience / project evidence:**")
        for line in row.relevant_experience:
            st.markdown(f"- {line}")
    st.info(row.explanation)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    source = st.radio("Candidate source", ["Upload PDF resumes", "Sample dataset (30 synthetic)", "Both"])
    st.subheader("Score weights")
    st.caption("Automatically normalised to sum to 1.")
    w_text = st.slider("Text similarity (TF-IDF)", 0.0, 1.0, 0.40, 0.05)
    w_skill = st.slider("Skill coverage", 0.0, 1.0, 0.45, 0.05)
    w_exp = st.slider("Experience fit", 0.0, 1.0, 0.15, 0.05)
    top_n = st.number_input("Show top N", 1, 20, 5)
    st.divider()
    st.caption("⚠️ Prototype. The score reflects text overlap with the JD only. "
               "It is not a measure of suitability and must support, not replace, human decisions. "
               "No personal attributes (age, gender, etc.) are extracted or used.")

# ------------------------------------------------------------------ main
st.title("📄 Smart Resume Screening & Candidate Matching")
st.write("Paste or upload a job description, add resumes, and get an explainable ranked shortlist.")

st.subheader("1. Job description")
jc1, jc2 = st.columns([3, 1])
with jc2:
    st.button("Load sample JD", on_click=load_sample_jd)
    st.file_uploader("…or upload JD (.txt / .pdf)", type=["txt", "pdf"], key="jd_file", on_change=load_jd_from_file)
with jc1:
    st.text_area("Job description text", key="jd_text", height=260,
                 placeholder="Paste the job description here, including requirements and nice-to-haves…")
if st.session_state.get("jd_file_error"):
    st.error(st.session_state["jd_file_error"])

st.subheader("2. Resumes")
uploads = []
if source in ("Upload PDF resumes", "Both"):
    uploads = st.file_uploader("Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True)
if source in ("Sample dataset (30 synthetic)", "Both"):
    st.caption("Using the bundled synthetic dataset (duplicates and missing values are cleaned automatically).")

if st.button("🔍 Screen candidates", type="primary"):
    jd_text = st.session_state.get("jd_text", "").strip()
    records, errors, clean_log = [], [], None

    if len(jd_text.split()) < 10:
        st.error("Please provide a job description (at least a few lines).")
    elif source == "Upload PDF resumes" and not uploads:
        st.error("Please upload at least one PDF resume, or switch to the sample dataset.")
    else:
        with st.spinner("Reading resumes and scoring candidates…"):
            if source in ("Sample dataset (30 synthetic)", "Both"):
                ds_records, clean_log = cached_sample_dataset()
                records += ds_records
            if uploads:
                pdf_records, errors = records_from_pdfs(uploads)
                records += pdf_records
            if not records:
                st.error("No readable resumes were found. See the reasons below.")
            else:
                ranked = rank_candidates(records, jd_text, Weights(w_text, w_skill, w_exp))
                st.session_state["results"] = {"ranked": ranked, "errors": errors, "clean_log": clean_log}
        if not records:
            for fname, reason in errors:
                st.warning(f"**{fname}** - {reason}")

res = st.session_state.get("results")
if res:
    ranked: pd.DataFrame = res["ranked"]
    st.divider()
    st.header("Results")

    if res["errors"]:
        with st.expander(f"⚠️ {len(res['errors'])} file(s) skipped", expanded=True):
            for fname, reason in res["errors"]:
                st.markdown(f"- **{fname}** - {reason}")
    if res["clean_log"]:
        with st.expander("Data cleaning log (sample dataset)"):
            st.json(res["clean_log"])

    jd = ranked.attrs.get("jd", {})
    wu = ranked.attrs.get("weights_used")
    with st.expander("How the job description was understood", expanded=False):
        st.markdown(f"**Required skills:** {', '.join(jd.get('required_skills', [])) or '_none detected_'}")
        st.markdown(f"**Nice-to-have skills:** {', '.join(jd.get('preferred_skills', [])) or '_none detected_'}")
        st.markdown(f"**Minimum experience:** {jd.get('min_years') or '_not stated_'} years")
        if wu:
            st.caption(f"Effective weights → text {wu.text:.2f}, skills {wu.skills:.2f}, experience {wu.experience:.2f} "
                       "(a component with no JD signal is dropped and the rest renormalised).")
        if not jd.get("required_skills") and not jd.get("preferred_skills"):
            st.warning("No known skills were detected in the JD, so ranking relies on text similarity only.")

    tab_top, tab_cmp, tab_all, tab_how = st.tabs(["🏆 Top matches", "⚖️ Compare two", "📋 All candidates & export", "ℹ️ How scoring works"])

    with tab_top:
        top = ranked.head(int(top_n))
        st.dataframe(
            top[["rank", "name", "match_score", "skill_match", "text_similarity", "experience_years"]],
            column_config={
                "match_score": st.column_config.ProgressColumn("Match score", min_value=0, max_value=100, format="%.1f"),
                "skill_match": st.column_config.NumberColumn("Skill match %", format="%.1f"),
                "text_similarity": st.column_config.NumberColumn("TF-IDF similarity %", format="%.1f"),
                "experience_years": st.column_config.NumberColumn("Years exp."),
            },
            hide_index=True,
        )
        for _, row in top.iterrows():
            with st.expander(f"#{row['rank']}  {row['name']}  -  {row['match_score']:.1f}", expanded=(row["rank"] == 1)):
                candidate_card(row)

    with tab_cmp:
        if len(ranked) < 2:
            st.info("Need at least two candidates to compare.")
        else:
            labels = [f"#{r['rank']} {r['name']}" for _, r in ranked.iterrows()]
            a = st.selectbox("Candidate A", labels, index=0)
            b = st.selectbox("Candidate B", labels, index=1)
            ca, cb = st.columns(2)
            for col, label in ((ca, a), (cb, b)):
                with col:
                    st.subheader(label)
                    candidate_card(ranked.iloc[labels.index(label)])

    with tab_all:
        flat = report_dataframe(ranked)
        st.dataframe(flat.drop(columns=["relevant_experience"]), hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", flat.to_csv(index=False).encode("utf-8"), "candidate_ranking.csv", "text/csv")
        d2.download_button("⬇️ Download Excel", to_excel_bytes(flat), "candidate_ranking.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_how:
        st.markdown("""
**Match score (0-100)** = weighted blend of three separately reported parts:

1. **Text similarity** - cosine similarity between TF-IDF vectors of the JD and the resume (uni- and bi-grams).
   Captures overall vocabulary overlap; typically low in absolute terms because resumes and JDs are written differently.
2. **Skill match** - share of JD skills found literally in the resume (nice-to-haves count half).
3. **Experience fit** - candidate years ÷ years requested, capped at 100%.

**What it does *not* measure:** actual ability, quality or depth of experience, soft skills, potential, culture fit,
or skills outside the built-in taxonomy. Skills marked *inferred* (e.g. pandas → Python) are shown for context but
are **not** counted as matches.
""")
