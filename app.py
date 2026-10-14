"""Streamlit app: paste a job posting, get a scam-risk score and the reasons why.

Run:  streamlit run app.py
"""
import html
import re
from pathlib import Path

import joblib
import streamlit as st

import jobscam

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

SAMPLES = {
    "Suspicious example": {
        "title": "Data Entry Work From Home",
        "company_profile": "",
        "description": "URGENT hiring! No experience needed. Earn $500 per week from home. Pay a registration "
                       "fee of Rs 2000 to get started. Contact us on WhatsApp or mail hr.jobs@gmail.com. "
                       "Limited seats, apply now!",
        "requirements": "",
        "benefits": "",
        "logo": False, "remote": True, "questions": False,
    },
    "Legitimate example": {
        "title": "Machine Learning Engineer",
        "company_profile": "We are a Bengaluru-based fintech building credit risk tools used by over 40 banks.",
        "description": "Design, train and deploy machine learning models for credit risk. You will work with data "
                       "scientists and backend engineers, own models from experiment to production, and monitor "
                       "drift in live systems. Our team reviews every model change together.",
        "requirements": "BE/BTech in CS, 2+ years of Python, scikit-learn and SQL, experience deploying models.",
        "benefits": "Health insurance, learning budget, flexible hours.",
        "logo": True, "remote": False, "questions": True,
    },
}


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "models" / "model.joblib"
    art = joblib.load(path)
    return art["model"], art["threshold"]


def load_sample(name):
    for k, v in SAMPLES[name].items():
        st.session_state[k] = v


st.title("🕵️ Fake Job Posting Detector")
st.caption("Paste a job or internship listing. The model scores how scam-like it looks and shows why. "
           "It is a screening aid, not a verdict: always verify the company independently.")

model, threshold = load_model()

c1, c2 = st.columns(2)
c1.button("Try a suspicious example", on_click=load_sample, args=("Suspicious example",), use_container_width=True)
c2.button("Try a legitimate example", on_click=load_sample, args=("Legitimate example",), use_container_width=True)

title = st.text_input("Job title", key="title")
company_profile = st.text_area("Company profile / About us", key="company_profile", height=90)
description = st.text_area("Job description", key="description", height=160)
requirements = st.text_area("Requirements", key="requirements", height=90)
benefits = st.text_area("Benefits", key="benefits", height=70)
k1, k2, k3 = st.columns(3)
logo = k1.checkbox("Listing shows a company logo", key="logo")
remote = k2.checkbox("Remote / work from home", key="remote")
questions = k3.checkbox("Has screening questions", key="questions")

if st.button("Analyse posting", type="primary", use_container_width=True):
    if not (title.strip() or description.strip()):
        st.warning("Add at least a job title or description.")
        st.stop()

    row = {"title": title, "company_profile": company_profile, "description": description,
           "requirements": requirements, "benefits": benefits,
           "has_company_logo": int(logo), "telecommuting": int(remote), "has_questions": int(questions)}

    p = jobscam.predict_proba(model, row)
    is_fake = p >= threshold
    words, flags = jobscam.explain(model, row)
    rules = jobscam.rule_flags(row)

    st.divider()
    if is_fake:
        st.error(f"⚠️ Likely a SCAM. Risk score {p:.0%}")
    elif p >= threshold * 0.5:
        st.warning(f"🟡 Some warning signs. Risk score {p:.0%}")
    else:
        st.success(f"✅ Looks legitimate. Risk score {p:.0%}")
    st.progress(min(p, 1.0))

    if rules:
        st.subheader("Red flags found")
        for r in rules:
            st.markdown(f"- 🚩 {r}")

    fraud_words = [(w, c) for w, c in words if c > 0][:8]
    safe_words = [(w, c) for w, c in sorted(words, key=lambda t: t[1])[:5] if c < 0]
    if fraud_words:
        st.subheader("Words that raised the score")
        st.markdown(" ".join(f"`{w}`" for w, _ in fraud_words))
    if safe_words:
        st.subheader("Words that lowered it")
        st.markdown(" ".join(f"`{w}`" for w, _ in safe_words))

    if description.strip() and fraud_words:
        singles = {w for w, _ in fraud_words if " " not in w}
        if singles:
            pat = re.compile(r"\b(" + "|".join(map(re.escape, singles)) + r")\b", re.I)
            marked = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", html.escape(description))
            st.subheader("Description with flagged words")
            st.markdown(marked, unsafe_allow_html=True)

    with st.expander("How the score is built (model feature contributions)"):
        labels = {"no_profile": "No company profile", "no_logo": "No company logo",
                  "no_requirements": "No requirements listed", "no_salary": "No salary listed",
                  "no_benefits": "No benefits listed", "no_exp_claim": "'No experience needed' claim",
                  "money_count": "Earning-promise language", "urgency_count": "Urgency language",
                  "exclaim_count": "Exclamation marks", "caps_ratio": "ALL-CAPS ratio",
                  "desc_len_log": "Description length", "n_missing_fields": "Number of empty fields",
                  "telecommuting": "Remote role", "has_questions": "Screening questions"}
        for name, c in flags[:8]:
            arrow = "↑ raises" if c > 0 else "↓ lowers"
            st.write(f"{arrow} risk: **{labels.get(name, name)}** ({c:+.2f})")
        st.caption("Contributions are exact for this linear model (coefficient × feature value, in log-odds).")
