"""
generate_data.py
----------------
Creates realistic sample datasets that follow the exact schema of the
Bank GoodCredit MySQL database (project_banking):

    Cust_Account       -> data/raw/customer_account.csv
    Cust_Enquiry       -> data/raw/customer_enquiry.csv
    Cust_Demographics  -> data/raw/customer_demographics.csv

Use this when the client database is not reachable. If you have database
access, run `python src/fetch_from_mysql.py` instead and it will overwrite
these files with the real tables.

The data is synthetic but statistically shaped like credit-bureau data:
  * ~4% bad rate (30+ DPD customers)
  * payment history strings made of 3-character monthly DPD codes
  * date columns in DD-Mon-YY format (as in the source tables)
  * obscured demographic columns feature_1 ... feature_79 with missing
    values, constant columns and high-cardinality columns
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

ACCT_TYPES = [10, 5, 6, 1, 2, 13, 17, 7, 3, 9]          # 10 = credit card
ACCT_TYPE_P = [0.30, 0.18, 0.20, 0.08, 0.05, 0.06, 0.04, 0.03, 0.03, 0.03]
ENQ_PURPOSES = [10, 5, 6, 1, 2, 13, 17, 3, 0, 14]
ENQ_PURPOSE_P = [0.32, 0.20, 0.17, 0.08, 0.05, 0.06, 0.04, 0.03, 0.03, 0.02]

PULL_DATE = datetime(2015, 11, 10)          # date the bureau data was pulled


def fmt(d):
    """DD-Mon-YY like the source tables (e.g. 10-Nov-15)."""
    return None if d is None else d.strftime("%d-%b-%y")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dpd_code(dpd, rng):
    if dpd < 0:
        return "XXX"
    if dpd == 0:
        return "STD" if rng.random() < 0.15 else "000"
    return f"{min(int(dpd), 999):03d}"


def make_payment_history(n_months, p_del, severity, rng):
    """Most recent month first. Returns (hist1, hist2) of 18 months each."""
    codes, state = [], 0
    for _ in range(n_months):
        if state > 0:                                  # already delinquent
            r = rng.random()
            if r < 0.45:
                state = 0                              # cured
            elif r < 0.45 + 0.35 * severity:
                state = min(state + 30, 180)           # rolled forward
        elif rng.random() < p_del:
            state = int(rng.choice([5, 12, 20, 28, 35, 45, 62],
                                   p=[.25, .2, .15, .1, .14, .09, .07]))
        dpd = state if state > 0 else 0
        if rng.random() < 0.04:
            dpd = -1                                   # no info that month
        codes.append(dpd_code(dpd, rng))
    s = "".join(codes)
    return s[:54], (s[54:108] if len(s) > 54 else None)


def build(n_customers=15000, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(RAW, exist_ok=True)

    cust_ids = np.arange(1, n_customers + 1) * 7 + 1000
    z = rng.normal(0, 1, n_customers)                  # latent credit risk
    p_bad = sigmoid(-3.55 + 1.15 * z)
    bad = (rng.random(n_customers) < p_bad).astype(int)

    # ------------------------------------------------------------------ #
    # Cust_Account
    # ------------------------------------------------------------------ #
    acc_rows = []
    for cid, zi in zip(cust_ids, z):
        n_acc = max(1, rng.poisson(np.exp(1.85 + 0.12 * zi)))
        upload_dt = PULL_DATE - timedelta(days=int(rng.integers(0, 20)))
        cust_util = sigmoid(-0.4 + 0.75 * zi + rng.normal(0, 0.5))
        for _ in range(n_acc):
            atype = int(rng.choice(ACCT_TYPES, p=ACCT_TYPE_P))
            age_days = int(rng.gamma(2.2, 420 * np.exp(-0.15 * zi))) + 30
            opened = PULL_DATE - timedelta(days=min(age_days, 5200))
            months = max(1, min(36, age_days // 30))
            closed = None
            if rng.random() < sigmoid(-0.6 - 0.2 * zi) and age_days > 200:
                closed = opened + timedelta(days=int(rng.integers(90, age_days)))
            reporting = (closed or PULL_DATE) - timedelta(days=int(rng.integers(0, 40)))
            gap = max(1, (reporting - opened).days)
            # risky customers stop paying earlier relative to account life
            frac = np.clip(rng.beta(6 - 1.4 * np.clip(zi, -2, 2.5), 1.6), 0.05, 1)
            last_pay = opened + timedelta(days=int(gap * frac))

            if atype == 10:
                climit = float(rng.choice([25e3, 40e3, 50e3, 75e3, 1e5, 1.5e5, 2e5, 3e5]))
                climit *= np.exp(-0.2 * zi)
                cash = climit * 0.2 if rng.random() < 0.55 else np.nan
                util = np.clip(cust_util + rng.normal(0, 0.15), 0, 1.3)
                cur_bal = round(climit * util, 0)
                high_credit = round(max(cur_bal, climit * rng.uniform(0.3, 1.1)), 0)
            else:
                climit, cash = np.nan, np.nan
                high_credit = round(float(np.exp(rng.normal(11.5, 1.1))), 0)
                left = np.clip(1 - frac + rng.normal(0, 0.15) + 0.1 * zi, 0, 1)
                cur_bal = 0.0 if closed else round(high_credit * left, 0)

            p_del = sigmoid(-3.6 + 1.05 * zi + rng.normal(0, 0.35))
            h1, h2 = make_payment_history(months, p_del, sigmoid(zi), rng)
            past_due = 0.0
            if h1[:3] not in ("000", "STD", "XXX") and cur_bal > 0:
                past_due = round(cur_bal * rng.uniform(0.03, 0.25), 0)

            acc_rows.append({
                "dt_opened": fmt(PULL_DATE),
                "customer_no": cid,
                "upload_dt": fmt(upload_dt),
                "acct_type": atype,
                "owner_indic": int(rng.choice([1, 2, 3, 4], p=[.86, .06, .03, .05])),
                "opened_dt": fmt(opened),
                "last_paymt_dt": fmt(last_pay) if rng.random() > 0.03 else None,
                "closed_dt": fmt(closed),
                "reporting_dt": fmt(reporting),
                "high_credit_amt": high_credit if rng.random() > 0.05 else np.nan,
                "cur_balance_amt": cur_bal,
                "amt_past_due": past_due if past_due > 0 else np.nan,
                "paymenthistory1": h1,
                "paymenthistory2": h2,
                "paymt_str_dt": fmt(reporting.replace(day=1)),
                "paymt_end_dt": fmt((reporting - timedelta(days=30 * (months - 1))).replace(day=1)),
                "creditlimit": round(climit, 0) if not np.isnan(climit) else np.nan,
                "cashlimit": round(cash, 0) if not np.isnan(cash) else np.nan,
                "rateofinterest": round(rng.uniform(9, 26), 2) if rng.random() < 0.12 else np.nan,
                "paymentfrequency": int(rng.choice([1, 3])) if rng.random() < 0.25 else np.nan,
                "actualpaymentamount": round(cur_bal * rng.uniform(0.02, 0.2), 0)
                if (rng.random() < 0.3 and cur_bal > 0) else np.nan,
            })
    acc = pd.DataFrame(acc_rows)

    # ------------------------------------------------------------------ #
    # Cust_Enquiry
    # ------------------------------------------------------------------ #
    enq_rows = []
    for cid, zi in zip(cust_ids, z):
        n_enq = rng.poisson(np.exp(2.0 + 0.32 * zi))
        recency_scale = 650 * np.exp(-0.3 * zi)         # risky = recent enquiries
        upload_dt = PULL_DATE - timedelta(days=int(rng.integers(0, 20)))
        for _ in range(n_enq):
            days_ago = int(min(rng.exponential(recency_scale), 3000)) + 1
            enq_rows.append({
                "dt_opened": fmt(PULL_DATE),
                "customer_no": cid,
                "upload_dt": fmt(upload_dt),
                "enquiry_dt": fmt(PULL_DATE - timedelta(days=days_ago)),
                "enq_purpose": int(rng.choice(ENQ_PURPOSES, p=ENQ_PURPOSE_P)),
                "enq_amt": int(round(float(np.exp(rng.normal(11.2 + 0.15 * zi, 1.2))), -2)) + 1000,
            })
    enq = pd.DataFrame(enq_rows)

    # ------------------------------------------------------------------ #
    # Cust_Demographics (feature_1 ... feature_79 obscured)
    # ------------------------------------------------------------------ #
    demo = pd.DataFrame({
        "dt_opened": fmt(PULL_DATE),
        "customer_no": cust_ids,
        "entry_time": [fmt(PULL_DATE - timedelta(days=int(d)))
                       for d in rng.integers(0, 90, n_customers)],
    })
    n = n_customers
    cat_pool = list("ABCDEFGH")
    for i in range(1, 80):
        col = f"feature_{i}"
        kind = i % 9
        signal = rng.normal(0, 1, n)
        if i in (3, 7, 14, 25, 33, 52, 61, 70):       # informative numeric
            w = rng.choice([-0.55, -0.4, 0.4, 0.55])
            vals = 50 + 12 * (w * z + np.sqrt(1 - w ** 2) * signal)
        elif i in (2, 19, 44):                         # informative categorical
            idx = np.clip(((0.6 * z + 0.8 * signal) * 1.3 + 3.5).astype(int), 0, 7)
            vals = np.array(cat_pool)[idx]
        elif i in (12, 38):                            # Y/N flag
            vals = np.where(rng.random(n) < sigmoid(-0.3 + 0.5 * z), "Y", "N")
        elif i in (5, 47):                             # constant columns
            vals = np.full(n, 1)
        elif i == 9:                                   # near-unique id-like
            vals = rng.integers(10 ** 7, 10 ** 8, n)
        elif kind == 0:                                # count-like noise
            vals = rng.poisson(3, n)
        elif kind in (1, 2):                           # continuous noise
            vals = np.round(np.exp(rng.normal(9, 1, n)), 2)
        elif kind == 3:                                # pincode-like categorical
            vals = rng.choice([f"{c}{k}" for c in cat_pool for k in range(6)], n)
        else:
            vals = np.round(rng.normal(100, 25, n), 2)
        s = pd.Series(vals)
        # missingness: some columns almost empty, most a little
        miss = 0.0
        if i in (15, 22, 29, 41, 56, 66, 73, 78):
            miss = rng.uniform(0.75, 0.97)
        elif i % 4 == 0:
            miss = rng.uniform(0.01, 0.2)
        if miss:
            s[rng.random(n) < miss] = np.nan
        demo[col] = s.values
    demo["Bad_label"] = bad

    acc.to_csv(os.path.join(RAW, "customer_account.csv"), index=False)
    enq.to_csv(os.path.join(RAW, "customer_enquiry.csv"), index=False)
    demo.to_csv(os.path.join(RAW, "customer_demographics.csv"), index=False)
    print(f"Cust_Account      : {acc.shape}")
    print(f"Cust_Enquiry      : {enq.shape}")
    print(f"Cust_Demographics : {demo.shape}  bad rate = {bad.mean():.3%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(a.customers, a.seed)
