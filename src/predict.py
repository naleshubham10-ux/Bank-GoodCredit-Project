"""
predict.py
----------
Scores customers with the trained model and converts the probability of
default into a cred score (300-900, higher = more creditworthy) and a
risk band.

Usage
    python src/predict.py                     # scores every customer in data/raw
    python src/predict.py --customer 1007     # one customer

Output -> reports/customer_cred_scores.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_feature_matrix  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "credit_score_model.pkl")
OUT = os.path.join(ROOT, "reports", "customer_cred_scores.csv")

# score scaling: 600 points at odds 19:1 (good:bad), +40 points doubles odds
BASE_SCORE, BASE_ODDS, PDO = 600, 19, 40


def prob_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return np.clip(np.round(offset + factor * np.log((1 - p) / p)), 300, 900).astype(int)


def risk_band(score):
    bins = [0, 500, 580, 650, 720, 901]
    labels = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
    return pd.cut(score, bins=bins, labels=labels, right=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", type=int, default=None)
    a = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model, feats = bundle["model"], bundle["features"]
    X, _ = build_feature_matrix(save=False, verbose=False)
    if a.customer is not None:
        X = X.loc[[a.customer]]
    p = model.predict_proba(X[feats])[:, 1]

    out = pd.DataFrame({"customer_no": X.index, "prob_bad": np.round(p, 4)})
    out["cred_score"] = prob_to_score(p)
    out["risk_band"] = risk_band(out["cred_score"])
    out["decision"] = np.where(out["cred_score"] >= 580, "Approve / maintain limit",
                               "Review / reduce exposure")

    if a.customer is not None:
        print(out.to_string(index=False))
    else:
        out.to_csv(OUT, index=False)
        print(f"Scored {len(out)} customers with {bundle['name']} -> {OUT}")
        print(out["risk_band"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
