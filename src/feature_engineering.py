"""
feature_engineering.py
----------------------
Turns the three raw tables into one customer-level feature matrix.

  Account table   -> payment-history DPD features, utilisation, balances,
                     account-age and payment-recency features
  Enquiry table   -> enquiry counts by recency window, amounts, purposes
  Demographics    -> cleaned feature_1..feature_79 (drop >70% missing,
                     constant and id-like columns; impute; encode)

Output: data/processed/feature_matrix.csv  (one row per customer_no)
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

DATE_FMT = "%d-%b-%y"
MISSING_THRESHOLD = 0.70


# ---------------------------------------------------------------------- #
# Loading
# ---------------------------------------------------------------------- #
def load_raw():
    acc = pd.read_csv(os.path.join(RAW, "customer_account.csv"),
                      dtype={"paymenthistory1": str, "paymenthistory2": str})
    enq = pd.read_csv(os.path.join(RAW, "customer_enquiry.csv"))
    demo = pd.read_csv(os.path.join(RAW, "customer_demographics.csv"), low_memory=False)
    # the source table sometimes calls it "customer no"
    demo = demo.rename(columns={"customer no": "customer_no"})
    return acc, enq, demo


def to_date(s):
    return pd.to_datetime(s, format=DATE_FMT, errors="coerce")


# ---------------------------------------------------------------------- #
# Payment history parsing
# ---------------------------------------------------------------------- #
def parse_history(h1, h2):
    """Return list of monthly DPD values (most recent first). NaN = unknown."""
    s = ""
    for h in (h1, h2):
        if isinstance(h, str):
            s += h.replace('"', "").strip()
    out = []
    for i in range(0, len(s) - len(s) % 3, 3):
        code = s[i:i + 3]
        if code.isdigit():
            out.append(int(code))
        elif code in ("STD", "000"):
            out.append(0)
        elif code == "SMA":
            out.append(60)
        elif code in ("SUB", "DBT", "LSS"):
            out.append(90)
        else:                       # XXX / garbage
            out.append(np.nan)
    return out


def account_history_stats(hist):
    a = np.array(hist, dtype=float)
    known = a[~np.isnan(a)]
    n = len(a)
    if len(known) == 0:
        return dict(ph_len=n, ph_known=0, cnt_0_29=0, cnt_30p=0, cnt_60p=0,
                    cnt_90p=0, max_dpd=0, first_30p_month=np.nan, mean_dpd=0)
    idx30 = np.where(a >= 30)[0]
    return dict(
        ph_len=n,
        ph_known=len(known),
        cnt_0_29=int(((known > 0) & (known < 30)).sum()),
        cnt_30p=int((known >= 30).sum()),
        cnt_60p=int((known >= 60).sum()),
        cnt_90p=int((known >= 90).sum()),
        max_dpd=float(known.max()),
        # months back (0 = latest) at which the most recent 30+ dpd appeared
        first_30p_month=float(idx30.min()) if len(idx30) else np.nan,
        mean_dpd=float(known.mean()),
    )


# ---------------------------------------------------------------------- #
# Account features
# ---------------------------------------------------------------------- #
def account_features(acc):
    acc = acc.copy()
    for c in ["dt_opened", "opened_dt", "last_paymt_dt", "closed_dt", "reporting_dt"]:
        acc[c] = to_date(acc[c])

    stats = pd.DataFrame([account_history_stats(parse_history(a, b))
                          for a, b in zip(acc["paymenthistory1"], acc["paymenthistory2"])])
    acc = pd.concat([acc.reset_index(drop=True), stats], axis=1)

    acc["diff_lastpaymt_opened"] = (acc["last_paymt_dt"] - acc["opened_dt"]).dt.days
    acc["acct_age_days"] = (acc["dt_opened"] - acc["opened_dt"]).dt.days
    acc["days_since_last_paymt"] = (acc["dt_opened"] - acc["last_paymt_dt"]).dt.days
    acc["is_open"] = acc["closed_dt"].isna().astype(int)
    acc["is_card"] = (acc["acct_type"] == 10).astype(int)
    acc["has_30p"] = (acc["cnt_30p"] > 0).astype(int)
    acc["has_0_29"] = (acc["cnt_0_29"] > 0).astype(int)
    acc["past_due_flag"] = (acc["amt_past_due"].fillna(0) > 0).astype(int)

    g = acc.groupby("customer_no")
    f = pd.DataFrame(index=g.size().index)
    f["n_accounts"] = g.size()
    f["n_open_accounts"] = g["is_open"].sum()
    f["n_closed_accounts"] = f["n_accounts"] - f["n_open_accounts"]
    f["n_credit_cards"] = g["is_card"].sum()
    f["n_acct_types"] = g["acct_type"].nunique()
    f["pct_open_accounts"] = f["n_open_accounts"] / f["n_accounts"]

    # balances / limits
    f["total_high_credit_amt"] = g["high_credit_amt"].sum()
    f["total_cur_balance_amt"] = g["cur_balance_amt"].sum()
    f["mean_cur_balance_amt"] = g["cur_balance_amt"].mean()
    f["total_amt_past_due"] = g["amt_past_due"].sum()
    f["n_accounts_past_due"] = g["past_due_flag"].sum()
    f["total_creditlimit"] = g["creditlimit"].sum()
    f["mean_creditlimit"] = g["creditlimit"].mean()
    f["mean_cashlimit"] = g["cashlimit"].mean()

    card = acc[acc["is_card"] == 1].groupby("customer_no")
    card_bal = card["cur_balance_amt"].sum()
    card_mean_bal = card["cur_balance_amt"].mean()
    f["Ratio_currbalance_creditlimit"] = (card_bal / f["total_creditlimit"].replace(0, np.nan))
    denom = card_mean_bal / (f["mean_creditlimit"] + f["mean_cashlimit"].fillna(0)).replace(0, np.nan)
    f["utilisation_trend"] = f["Ratio_currbalance_creditlimit"] / denom.replace(0, np.nan)
    f["ratio_balance_high_credit"] = f["total_cur_balance_amt"] / f["total_high_credit_amt"].replace(0, np.nan)
    f["ratio_past_due_balance"] = f["total_amt_past_due"].fillna(0) / f["total_cur_balance_amt"].replace(0, np.nan)

    # dates
    f["total_diff_lastpaymt_opened_dt"] = g["diff_lastpaymt_opened"].sum()
    f["mean_diff_lastpaymt_opened_dt"] = g["diff_lastpaymt_opened"].mean()
    f["mean_acct_age_days"] = g["acct_age_days"].mean()
    f["max_acct_age_days"] = g["acct_age_days"].max()
    f["min_acct_age_days"] = g["acct_age_days"].min()
    f["min_days_since_last_paymt"] = g["days_since_last_paymt"].min()
    f["mean_days_since_last_paymt"] = g["days_since_last_paymt"].mean()

    # payment history
    f["payment_history_avg_dpd_0_29_bucket"] = g["cnt_0_29"].mean()
    f["payment_history_total_dpd_0_29"] = g["cnt_0_29"].sum()
    f["payment_history_total_30p"] = g["cnt_30p"].sum()
    f["payment_history_total_60p"] = g["cnt_60p"].sum()
    f["payment_history_total_90p"] = g["cnt_90p"].sum()
    f["n_accounts_ever_30p"] = g["has_30p"].sum()
    f["n_accounts_ever_0_29"] = g["has_0_29"].sum()
    f["pct_accounts_ever_30p"] = f["n_accounts_ever_30p"] / f["n_accounts"]
    f["max_dpd_ever"] = g["max_dpd"].max()
    f["mean_dpd"] = g["mean_dpd"].mean()
    f["min_months_last_30_plus"] = g["first_30p_month"].min()
    f["payment_history_mean_length"] = g["ph_len"].mean()
    f["pct_history_known"] = g["ph_known"].sum() / g["ph_len"].sum().replace(0, np.nan)
    # never-30+ customers: fill with a value beyond the 36-month window
    f["min_months_last_30_plus"] = f["min_months_last_30_plus"].fillna(99)
    return f


# ---------------------------------------------------------------------- #
# Enquiry features
# ---------------------------------------------------------------------- #
def enquiry_features(enq):
    enq = enq.copy()
    enq["dt_opened"] = to_date(enq["dt_opened"])
    enq["enquiry_dt"] = to_date(enq["enquiry_dt"])
    enq["days_ago"] = (enq["dt_opened"] - enq["enquiry_dt"]).dt.days

    g = enq.groupby("customer_no")
    f = pd.DataFrame(index=g.size().index)
    f["count_enquiry"] = g.size()
    for d in (30, 90, 180, 365):
        f[f"count_enquiry_recency_{d}"] = enq[enq["days_ago"] <= d].groupby("customer_no").size()
    f["mean_diff_open_enquiry_dt"] = g["days_ago"].mean()
    f["min_days_since_enquiry"] = g["days_ago"].min()
    f["max_days_since_enquiry"] = g["days_ago"].max()
    f["total_enq_amt"] = g["enq_amt"].sum()
    f["mean_enq_amt"] = g["enq_amt"].mean()
    f["max_enq_amt"] = g["enq_amt"].max()
    f["n_enq_purposes"] = g["enq_purpose"].nunique()
    f["count_enq_credit_card"] = enq[enq["enq_purpose"] == 10].groupby("customer_no").size()
    f["count_enq_personal_loan"] = enq[enq["enq_purpose"] == 5].groupby("customer_no").size()
    f["enquiry_recency_ratio_90_365"] = f["count_enquiry_recency_90"] / f["count_enquiry_recency_365"]
    count_cols = [c for c in f.columns if c.startswith("count_enq")]
    f[count_cols] = f[count_cols].fillna(0)
    return f


# ---------------------------------------------------------------------- #
# Demographic cleaning
# ---------------------------------------------------------------------- #
def clean_demographics(demo, report=None):
    """Drop unusable columns, impute, encode. `report` collects decisions."""
    report = report if report is not None else {}
    d = demo.drop(columns=["dt_opened", "entry_time"], errors="ignore").copy()
    feats = [c for c in d.columns if c.startswith("feature_")]

    miss = d[feats].isna().mean()
    drop_missing = miss[miss > MISSING_THRESHOLD].index.tolist()
    nunique = d[feats].nunique(dropna=True)
    drop_constant = nunique[nunique <= 1].index.tolist()
    # id-like = integer or text column where almost every value is unique
    # (continuous floats are naturally unique, so they are not id-like)
    drop_idlike = [c for c in feats
                   if nunique[c] > 0.9 * d[c].notna().sum()
                   and not pd.api.types.is_float_dtype(d[c])
                   and c not in drop_missing]
    drop = sorted(set(drop_missing + drop_constant + drop_idlike),
                  key=lambda x: int(x.split("_")[1]))
    d = d.drop(columns=drop)
    report.update(drop_missing=drop_missing, drop_constant=drop_constant,
                  drop_idlike=drop_idlike)

    feats = [c for c in d.columns if c.startswith("feature_")]
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(d[c])]
    num_cols = [c for c in feats if c not in cat_cols]
    for c in num_cols:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    for c in cat_cols:
        d[c] = d[c].fillna("MISSING")
        if d[c].nunique() == 2 and set(d[c].unique()) <= {"Y", "N"}:
            d[c] = (d[c] == "Y").astype(int)
        else:
            # frequency encoding: uses no target information, so it cannot
            # leak Bad_label into the features before the train/test split
            freq = d[c].value_counts(normalize=True)
            d[c] = d[c].map(freq).astype(float)        # frequency encoding
    report.update(categorical=cat_cols, numeric=num_cols)
    return d.set_index("customer_no"), report


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def build_feature_matrix(save=True, verbose=True):
    acc, enq, demo = load_raw()
    report = {}
    fa = account_features(acc)
    fe = enquiry_features(enq)
    fd, report = clean_demographics(demo, report)

    X = fd.join(fa, how="left").join(fe, how="left")
    # customers with no enquiries -> counts are zero
    enq_counts = [c for c in fe.columns if c.startswith("count_enq")]
    X[enq_counts] = X[enq_counts].fillna(0)
    X = X.replace([np.inf, -np.inf], np.nan)

    if save:
        os.makedirs(PROC, exist_ok=True)
        X.to_csv(os.path.join(PROC, "feature_matrix.csv"))
    if verbose:
        print(f"Feature matrix: {X.shape}  (account {fa.shape[1]}, "
              f"enquiry {fe.shape[1]}, demographic {fd.shape[1] - 1})")
        print(f"Dropped demographic columns: {len(report['drop_missing'])} missing>70%, "
              f"{len(report['drop_constant'])} constant, {len(report['drop_idlike'])} id-like")
    return X, report


if __name__ == "__main__":
    build_feature_matrix()
