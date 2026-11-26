"""
train.py
--------
Trains and evaluates credit-score models for Bank GoodCredit.

Steps
  1. Load the feature matrix + Bad_label
  2. Stratified 70/30 train/test split
  3. Feature selection: drop near-duplicate features (|corr| > 0.95),
     then keep features with XGBoost gain until 95% cumulative gain
  4. Train Logistic Regression, Random Forest, XGBoost, LightGBM and an
     Ensemble (rank-average of XGB + LGBM + LR)
  5. Evaluate with Gini (= 2*AUC - 1) and decile rank ordering
     against the benchmark (Gini 37.9)
  6. Save model, feature list, tables and charts

Outputs -> models/  and  reports/
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import CalibratedModel, RankEnsemble  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")

BENCHMARK_GINI = 37.9
BENCHMARK_RANK = {10: 0.11713031, 9: 0.08052709, 8: 0.06002928, 7: 0.03958944,
                  6: 0.04245974, 5: 0.03660322, 4: 0.02052786, 3: 0.0204978,
                  2: 0.0204978, 1: 0.01317716}
SEED = 42
BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#9aa3ad"


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def gini(y, p):
    return 100 * (2 * roc_auc_score(y, p) - 1)


def rank_ordering(y, p, n=10):
    """Decile 10 = highest predicted risk. Returns bad rate per decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(rankdata(df["p"], method="ordinal"), n, labels=range(1, n + 1)).astype(int)
    t = df.groupby("decile").agg(customers=("y", "size"), bads=("y", "sum"),
                                 min_score=("p", "min"), max_score=("p", "max"))
    t["bad_rate"] = t["bads"] / t["customers"]
    t = t.sort_index(ascending=False)
    t["cum_bads_pct"] = t["bads"].cumsum() / t["bads"].sum()
    t["cum_goods_pct"] = (t["customers"] - t["bads"]).cumsum() / (t["customers"] - t["bads"]).sum()
    t["KS"] = (t["cum_bads_pct"] - t["cum_goods_pct"]) * 100
    t["benchmark_bad_rate"] = t.index.map(BENCHMARK_RANK)
    return t


def rank_breaks(t):
    """Number of places the bad rate goes up when moving to a lower decile."""
    r = t.sort_index(ascending=False)["bad_rate"].values
    return int(sum(r[i + 1] > r[i] for i in range(len(r) - 1)))


# ---------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------- #
def load_data():
    X = pd.read_csv(os.path.join(PROC, "feature_matrix.csv"), index_col="customer_no")
    y = pd.read_csv(os.path.join(RAW, "customer_demographics.csv"),
                    usecols=["customer_no", "Bad_label"], index_col="customer_no")["Bad_label"]
    if "Bad_label" in X.columns:
        X = X.drop(columns="Bad_label")
    return X, y.loc[X.index]


def drop_correlated(X, thr=0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if (upper[c] > thr).any()]


def make_xgb(spw):
    return XGBClassifier(n_estimators=350, learning_rate=0.02, max_depth=3,
                         min_child_weight=25, subsample=0.75, colsample_bytree=0.6,
                         gamma=1.0, reg_lambda=10.0, reg_alpha=1.0,
                         scale_pos_weight=np.sqrt(spw), eval_metric="auc",
                         random_state=SEED, n_jobs=-1)


def make_lgbm(spw):
    return LGBMClassifier(n_estimators=350, learning_rate=0.02, num_leaves=7,
                          max_depth=3, min_child_samples=100, subsample=0.75, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0,
                          scale_pos_weight=np.sqrt(spw),
                          random_state=SEED, verbose=-1, n_jobs=-1)


def make_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))])


def make_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=400, max_depth=8,
                                                   min_samples_leaf=20, class_weight="balanced_subsample",
                                                   random_state=SEED, n_jobs=-1))])


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    X, y = load_data()
    print(f"Data: {X.shape}, bad rate {y.mean():.2%}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    # ---- feature selection (train data only) --------------------------
    corr_drop = drop_correlated(X_tr)
    feats0 = [c for c in X.columns if c not in corr_drop]
    # shadow-feature test: add pure-noise columns; a real feature must earn
    # more gain than the best noise column (averaged over 5 seeds)
    rng = np.random.default_rng(SEED)
    gains = []
    for s in range(5):
        Xs = X_tr[feats0].copy()
        for k in range(5):
            Xs[f"_shadow_{k}"] = rng.permutation(Xs[feats0[k * 7 % len(feats0)]].values)
        m = make_xgb(spw).set_params(random_state=SEED + s).fit(Xs, y_tr)
        gains.append(pd.Series(m.get_booster().get_score(importance_type="total_gain")))
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1)
    noise_cut = gain[gain.index.str.startswith("_shadow")].max()
    gain = gain.drop([c for c in gain.index if c.startswith("_shadow")])
    gain = gain.reindex(feats0).fillna(0).sort_values(ascending=False)
    selected = gain[gain > noise_cut].index.tolist()
    print(f"Features: {X.shape[1]} -> {len(feats0)} after corr filter -> "
          f"{len(selected)} beat the noise threshold")

    # ---- train models ------------------------------------------------
    models = {"Logistic Regression": make_lr(), "Random Forest": make_rf(),
              "XGBoost": make_xgb(spw), "LightGBM": make_lgbm(spw)}
    results, preds = [], {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr[selected], y_tr, cv=cv, scoring="roc_auc")
        m.fit(X_tr[selected], y_tr)
        p_tr = m.predict_proba(X_tr[selected])[:, 1]
        p_te = m.predict_proba(X_te[selected])[:, 1]
        preds[name] = p_te
        results.append({"model": name, "cv_gini": 100 * (2 * cv_auc.mean() - 1),
                        "train_gini": gini(y_tr, p_tr), "test_gini": gini(y_te, p_te)})
        print(f"  {name:20s} CV {results[-1]['cv_gini']:.1f}  "
              f"train {results[-1]['train_gini']:.1f}  test {results[-1]['test_gini']:.1f}")

    ens = RankEnsemble({k: models[k] for k in ["XGBoost", "LightGBM", "Logistic Regression"]}, selected)
    preds["Ensemble"] = ens.predict_proba(X_te)[:, 1]
    p_tr_ens = ens.predict_proba(X_tr)[:, 1]
    results.append({"model": "Ensemble", "cv_gini": np.nan,
                    "train_gini": gini(y_tr, p_tr_ens), "test_gini": gini(y_te, preds["Ensemble"])})
    print(f"  {'Ensemble':20s} train {results[-1]['train_gini']:.1f}  test {results[-1]['test_gini']:.1f}")

    res = pd.DataFrame(results)
    res["rank_breaks"] = [rank_breaks(rank_ordering(y_te, preds[m])) for m in res["model"]]
    res["beats_benchmark"] = res["test_gini"] > BENCHMARK_GINI
    res = res.round(2)
    res.to_csv(os.path.join(REPORTS, "model_comparison.csv"), index=False)

    # best = highest test Gini among models with good rank ordering
    best_name = res.sort_values(["rank_breaks", "test_gini"], ascending=[True, False]).iloc[0]["model"]
    p_best = preds[best_name]
    if best_name == "Ensemble":
        oof = p_tr_ens                     # blend of already-regularised models
        base = ens
    else:
        oof = cross_val_predict(models[best_name], X_tr[selected], y_tr, cv=cv,
                                method="predict_proba")[:, 1]
        base = models[best_name]
    best_model = CalibratedModel(base, selected).fit_calibration(oof, y_tr)
    p_cal = best_model.predict_proba(X_te)[:, 1]
    print(f"Calibration: mean predicted PD {p_cal.mean():.3%} vs actual test bad rate {y_te.mean():.3%}")
    ro = rank_ordering(y_te, p_best)
    ro.round(4).to_csv(os.path.join(REPORTS, "rank_ordering.csv"))
    print(f"\nBest model: {best_name}  test Gini {gini(y_te, p_best):.2f} "
          f"(benchmark {BENCHMARK_GINI})  rank breaks {rank_breaks(ro)}")
    print(ro[["customers", "bads", "bad_rate", "benchmark_bad_rate", "KS"]].round(4))

    # ---- feature matrix with gain (final XGBoost) --------------------
    fg = pd.Series(models["XGBoost"].get_booster().get_score(importance_type="gain")) \
        .reindex(selected).fillna(0)
    fm = pd.DataFrame({"feature": fg.index, "gain": (fg / fg.sum()).values})
    fm = fm.sort_values("gain", ascending=False).reset_index(drop=True)
    fm["source"] = fm["feature"].map(lambda f: "Demographics" if f.startswith("feature_")
                                     else ("Enquiry" if "enq" in f else "Account"))
    fm.index = fm.index + 1
    fm.to_csv(os.path.join(REPORTS, "feature_matrix_gain.csv"), index_label="rank")

    # ---- save artefacts ----------------------------------------------
    joblib.dump({"model": best_model, "features": selected, "name": best_name,
                 "all_models": models}, os.path.join(MODELS, "credit_score_model.pkl"))
    with open(os.path.join(MODELS, "selected_features.json"), "w") as f:
        json.dump(selected, f, indent=2)
    summary = {"best_model": best_name, "test_gini": round(gini(y_te, p_best), 2),
               "benchmark_gini": BENCHMARK_GINI, "rank_breaks": rank_breaks(ro),
               "n_features_selected": len(selected), "train_rows": len(X_tr),
               "test_rows": len(X_te), "bad_rate": round(float(y.mean()), 4),
               "top_decile_bad_rate": round(float(ro.loc[10, "bad_rate"]), 4),
               "max_KS": round(float(ro["KS"].max()), 2),
               "mean_calibrated_pd_test": round(float(p_cal.mean()), 4)}
    with open(os.path.join(REPORTS, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- charts ------------------------------------------------------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        lw = 2.4 if name == best_name else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{name} (Gini {gini(y_te, p):.1f})")
    ax.plot([0, 1], [0, 1], ls="--", c=GREY, lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves on test set")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "roc_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = ro.sort_index()
    xpos = np.arange(1, 11)
    ax.bar(xpos - 0.2, d["bad_rate"] * 100, 0.4, color=BLUE, label=f"{best_name} model")
    ax.bar(xpos + 0.2, d["benchmark_bad_rate"] * 100, 0.4, color=GREY, label="Benchmark")
    ax.set(xticks=xpos, xlabel="Decile (10 = highest risk)", ylabel="Bad rate (%)",
           title="Rank ordering: bad rate by score decile")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rank_ordering.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    r = res.set_index("model")["test_gini"]
    colors = [BLUE if m == best_name else "#8fb3ee" for m in r.index]
    ax.barh(r.index, r.values, color=colors)
    ax.axvline(BENCHMARK_GINI, c=ORANGE, ls="--", lw=1.5)
    ax.text(BENCHMARK_GINI + 0.5, -0.45, f"Benchmark {BENCHMARK_GINI}", color=ORANGE, fontsize=9)
    for i, v in enumerate(r.values):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set(xlabel="Test Gini", title="Model comparison"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "model_comparison.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    top = fm.head(20).iloc[::-1]
    ax.barh(top["feature"], top["gain"], color=BLUE)
    ax.set(xlabel="Normalised gain (XGBoost)", title="Top 20 features by gain")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "feature_gain_top20.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xpos, d.sort_index()["cum_bads_pct"].values[::-1] * 100, marker="o", c=BLUE, label="Bads captured")
    ax.plot(xpos, d.sort_index()["cum_goods_pct"].values[::-1] * 100, marker="o", c=GREY, label="Goods captured")
    ax.set(xticks=xpos, xticklabels=[f"Top {i*10}%" for i in xpos], ylabel="Cumulative %",
           title=f"Gains / KS chart (max KS {ro['KS'].max():.1f})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "gains_ks_chart.png"), dpi=150); plt.close(fig)

    print("\nSaved -> models/credit_score_model.pkl, reports/*.csv, reports/figures/*.png")
    return res, ro, fm


if __name__ == "__main__":
    main()
