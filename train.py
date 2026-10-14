"""Train and evaluate the fake job posting detector.

Usage:  python train.py
Outputs: models/model.joblib, models/metrics.json, results.md, figures/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score,
                             f1_score, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import jobscam

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
ROOT = Path(__file__).parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "fake_job_postings.csv")
y = df["fraudulent"].values
print(f"{len(df):,} postings, {y.sum():,} fraudulent ({y.mean():.1%})")

train_df, test_df, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte = jobscam.make_frame(train_df), jobscam.make_frame(test_df)


def scores(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "pr_auc": average_precision_score(y_true, proba),
            "threshold": float(thr)}


results, models = {}, {}
for name, use_flags in [("TF-IDF + LogReg (baseline)", False),
                        ("TF-IDF + scam features + LogReg", True)]:
    model = jobscam.build_pipeline(use_flags=use_flags)
    # Pick the decision threshold from cross-validated predictions on TRAIN only (no test leakage)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_train, oof)
    f1s = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[int(np.argmax(f1s))])
    cv_f1 = float(f1s.max())

    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    results[name] = {"default_0.5": scores(y_test, proba, 0.5),
                     "tuned": scores(y_test, proba, thr),
                     "cv_f1_train": cv_f1}
    models[name] = (model, proba, thr)
    print(f"\n{name}\n  test @0.5   : {results[name]['default_0.5']}"
          f"\n  test @{thr:.2f}: {results[name]['tuned']}\n  train CV F1 at tuned thr: {cv_f1:.3f}")

# ---- Final model = the one with features; save with its tuned threshold ----
final_name = "TF-IDF + scam features + LogReg"
final_model, final_proba, final_thr = models[final_name]
joblib.dump({"model": final_model, "threshold": final_thr}, ROOT / "models" / "model.joblib")
(ROOT / "models" / "metrics.json").write_text(json.dumps(results, indent=2))

# ---- Figures ----
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, (final_proba >= final_thr).astype(int),
    display_labels=["Real", "Fake"], cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Confusion matrix (test set)")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "confusion_matrix.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
for name, (_, proba, _) in models.items():
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax)
ax.set_title("Precision-Recall (fake class)")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "pr_curve.png", dpi=150); plt.close(fig)

# ---- Top words (global interpretability) ----
feats = final_model.named_steps["features"].get_feature_names_out()
coef = final_model.named_steps["clf"].coef_[0]
order = np.argsort(coef)
top_fake = [(feats[i].split("__", 1)[1], float(coef[i])) for i in order[::-1][:15]]
top_real = [(feats[i].split("__", 1)[1], float(coef[i])) for i in order[:15]]

# ---- results.md for the README ----
lines = ["| Model | Precision | Recall | F1 | PR-AUC |", "|---|---|---|---|---|"]
for name, res in results.items():
    s = res["tuned"]
    lines.append(f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['pr_auc']:.3f} |")
md = ["## Test-set results (fake class, 20% stratified hold-out, threshold tuned on train CV)\n",
      *lines, "",
      f"Test set: {len(y_test):,} postings ({int(y_test.sum())} fake). Final threshold: {final_thr:.2f}.", "",
      "## Strongest fraud signals (model coefficients)\n",
      ", ".join(f"`{w}`" for w, _ in top_fake), "",
      "## Strongest legitimate signals\n",
      ", ".join(f"`{w}`" for w, _ in top_real), ""]
(ROOT / "results.md").write_text("\n".join(md))
print("\nSaved model, metrics.json, results.md, figures/")
