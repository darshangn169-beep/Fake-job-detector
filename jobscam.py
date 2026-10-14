"""Shared logic: text cleaning, feature engineering, model pipeline, explanations.

Used by both train.py and app.py so training and serving never drift apart.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+|#url_[a-z0-9]+#)", re.I)
_EMAIL = re.compile(r"(\S+@\S+\.\S+|#email_[a-z0-9]+#)", re.I)
_PHONE = re.compile(r"(#phone_[a-z0-9]+#|\+?\d[\d\-\s]{8,}\d)", re.I)
_NONALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip HTML / URLs / emails / phone numbers / punctuation."""
    if not isinstance(text, str):
        return ""
    t = _HTML.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _PHONE.sub(" ", t)
    t = _NONALPHA.sub(" ", t.lower())
    return _SPACES.sub(" ", t).strip()


def _join_raw(row) -> str:
    # Missing values arrive as NaN (a truthy float), so test for str explicitly
    return " ".join(v for v in (row.get(f) for f in TEXT_FIELDS) if isinstance(v, str))


# --------------------------------------------------------------------------
# Scam-signal features (learned by the model)
# --------------------------------------------------------------------------
URGENCY = re.compile(
    r"\b(apply now|urgent(ly)?|immediate(ly)?|hiring now|limited (seats|slots|positions)|"
    r"act fast|asap|don'?t miss|hurry|today only)\b", re.I)
MONEY = re.compile(
    r"(earn \$?\d+|\$\d+\s*(/|per)\s*(hour|hr|day|week)|easy money|"
    r"unlimited (income|earning)|weekly pay|guaranteed (income|salary|pay))", re.I)
NO_EXP = re.compile(r"no (prior )?(experience|skills?|qualification)s?\s*(needed|required|necessary)?", re.I)

FLAG_COLS = [
    "desc_len_log", "caps_ratio", "exclaim_count", "urgency_count", "money_count",
    "no_exp_claim", "no_profile", "no_logo", "no_requirements", "no_salary",
    "no_benefits", "telecommuting", "has_questions", "n_missing_fields",
]


def _empty(x) -> bool:
    return (not isinstance(x, str)) or x.strip() == ""


def engineer_row(row) -> dict:
    desc = row.get("description", "")
    desc = desc if isinstance(desc, str) else ""
    raw = _join_raw(row)
    letters = [c for c in desc if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    missing_fields = sum(_empty(row.get(f)) for f in
                         ["company_profile", "requirements", "benefits", "salary_range",
                          "employment_type", "required_experience", "required_education",
                          "industry", "function", "department"])
    return {
        "desc_len_log": float(np.log1p(len(desc))),
        "caps_ratio": caps_ratio,
        "exclaim_count": raw.count("!"),
        "urgency_count": len(URGENCY.findall(raw)),
        "money_count": len(MONEY.findall(raw)),
        "no_exp_claim": int(bool(NO_EXP.search(raw))),
        "no_profile": int(_empty(row.get("company_profile"))),
        "no_logo": int(not bool(row.get("has_company_logo", 0))),
        "no_requirements": int(_empty(row.get("requirements"))),
        "no_salary": int(_empty(row.get("salary_range"))),
        "no_benefits": int(_empty(row.get("benefits"))),
        "telecommuting": int(bool(row.get("telecommuting", 0))),
        "has_questions": int(bool(row.get("has_questions", 0))),
        "n_missing_fields": missing_fields,
    }


def make_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw job-posting rows -> model input frame (cleaned text + engineered features)."""
    rows = df.to_dict("records")
    out = pd.DataFrame([engineer_row(r) for r in rows], index=df.index)
    out.insert(0, "text", [clean_text(_join_raw(r)) for r in rows])
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline(use_flags: bool = True, C: float = 4.0) -> Pipeline:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=60000,
                            sublinear_tf=True, stop_words="english")
    transformers = [("tfidf", tfidf, "text")]
    if use_flags:
        transformers.append(("flags", StandardScaler(with_mean=False), FLAG_COLS))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")),
    ])


def predict_proba(model: Pipeline, row: dict) -> float:
    frame = make_frame(pd.DataFrame([row]))
    return float(model.predict_proba(frame)[0, 1])


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def explain(model: Pipeline, row: dict, top_k: int = 8):
    """Per-feature contribution to the fraud score (coef x scaled value).

    For a linear model this is an exact decomposition of the log-odds,
    so no approximation (LIME/SHAP sampling) is needed.
    Returns (words, flags): lists of (name, contribution), most fraud-leaning first.
    """
    frame = make_frame(pd.DataFrame([row]))
    feats = model.named_steps["features"]
    clf = model.named_steps["clf"]
    x = feats.transform(frame)
    names = feats.get_feature_names_out()
    contrib = np.asarray(x.multiply(clf.coef_[0]).todense()).ravel() if hasattr(x, "multiply") \
        else x.ravel() * clf.coef_[0]
    items = [(n, c) for n, c in zip(names, contrib) if c != 0]
    words = [(n.split("__", 1)[1], c) for n, c in items if n.startswith("tfidf__")]
    flags = [(n.split("__", 1)[1], c) for n, c in items if n.startswith("flags__")]
    words.sort(key=lambda t: -t[1])
    flags.sort(key=lambda t: -t[1])
    return words, flags


# --------------------------------------------------------------------------
# Rule-based red flags (NOT learned: the public dataset anonymises emails/URLs,
# so the model cannot learn these; they are shown in the app as extra checks)
# --------------------------------------------------------------------------
_FREE_MAIL = re.compile(r"@(gmail|yahoo|ymail|hotmail|outlook|rediffmail|protonmail)\.", re.I)
_FEE = re.compile(r"(registration|processing|training|security|application|admin)\s+(fee|deposit)|"
                  r"pay\s+(a\s+)?(fee|deposit)|wire transfer|western union|money ?gram|"
                  r"(send|share).{0,20}(bank|card|aadhaar|passport)", re.I)
_MESSENGER = re.compile(r"(whatsapp|telegram|signal)\b", re.I)


def rule_flags(row: dict) -> list:
    raw = _join_raw(row)
    flags = []
    if _FREE_MAIL.search(raw):
        flags.append("Contact uses a free email domain (Gmail/Yahoo/etc.) instead of a company domain")
    if _FEE.search(raw):
        flags.append("Asks for a fee, deposit, or sensitive documents. Real employers do not charge candidates")
    if _MESSENGER.search(raw):
        flags.append("Asks you to continue on WhatsApp/Telegram")
    if URGENCY.search(raw):
        flags.append("Pressure language (\"apply now\", \"limited seats\", \"urgent\")")
    if MONEY.search(raw):
        flags.append("Unrealistic or vague earning promises")
    if NO_EXP.search(raw) and MONEY.search(raw):
        flags.append("\"No experience needed\" combined with high pay")
    if _empty(row.get("company_profile")):
        flags.append("No company profile provided")
    if len(str(row.get("description", ""))) < 200:
        flags.append("Very short, vague description")
    return flags
