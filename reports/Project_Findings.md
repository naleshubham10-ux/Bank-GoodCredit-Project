# PRCL-0015 Bank GoodCredit — Project Findings

## Goal 1 — Data exploration: what we found and what we decided

| # | Finding | Decision |
|---|---|---|
| 1 | Target is imbalanced: 5.05% bad (757 of 15,000 customers) | Use Gini/AUC, not accuracy. Stratify the split and use class weights |
| 2 | Account table has 1–19 accounts per customer (median 6) | Aggregate to one row per customer |
| 3 | Payment history is stored as 3-character monthly codes (`000`, `STD`, `005`…`999`, `XXX`), up to 36 months across two columns | Parse into monthly DPD. Build 0-29 / 30+ / 60+ / 90+ counts, max and mean DPD, and months since the last 30+ |
| 4 | Customers with no account ever 30+ DPD have a 1.7% bad rate, against 6.2% for the rest | Make delinquency history the core of the model |
| 5 | Bad rate rises with credit-card utilisation (top utilisation quintile ≈ 12%) | Build `Ratio_currbalance_creditlimit` and `utilisation_trend` |
| 6 | Bad rate rises with enquiries in the last 365 days (top quintile ≈ 12% vs ≈ 1.7% bottom) | Build enquiry counts for 30/90/180/365 days |
| 7 | `amt_past_due` (96% missing), `rateofinterest` (88%), `cashlimit` (84%), `actualpaymentamount` (80%) and `paymentfrequency` (75%) are mostly empty | Don't use them raw. Use flags and aggregates only |
| 8 | 15% of customers have no credit-card account, so their utilisation is undefined | Leave missing. Trees handle it natively and Logistic Regression imputes the median |
| 9 | Demographics: 8 columns > 70% missing, 2 constant, 1 ID-like (near-unique integer) | Drop the 11 columns |
| 10 | Demographics: several text-category columns and Y/N flags | Frequency-encode the categories (no target leakage). Map Y/N to 1/0 |
| 11 | Dates are text in `DD-Mon-YY` format | Parse them, then build durations: account age, last-payment gap, enquiry recency |

## Goal 2 — Feature matrix (selected features with gain)

The candidate list had 121 features: 38 account, 15 enquiry and 68 demographic. Of these, **50 were selected** because their gain beat random-noise shadow features.

| Source | Features selected | Share of total gain |
|---|---|---|
| Account | 15 | 49.5% |
| Demographics | 27 | 35.0% |
| Enquiry | 8 | 15.6% |

Top 15 by normalised XGBoost gain:

| Rank | Feature | Gain | Source |
|---|---|---|---|
| 1 | mean_dpd | 0.1151 | Account |
| 2 | n_accounts_ever_30p | 0.1084 | Account |
| 3 | pct_accounts_ever_30p | 0.0499 | Account |
| 4 | payment_history_avg_dpd_0_29_bucket | 0.0345 | Account |
| 5 | payment_history_total_dpd_0_29 | 0.0311 | Account |
| 6 | count_enquiry_recency_365 | 0.0302 | Enquiry |
| 7 | Ratio_currbalance_creditlimit | 0.0254 | Account |
| 8 | ratio_balance_high_credit | 0.0224 | Account |
| 9 | count_enquiry | 0.0223 | Enquiry |
| 10 | count_enquiry_recency_180 | 0.0220 | Enquiry |
| 11 | total_enq_amt | 0.0220 | Enquiry |
| 12 | count_enq_credit_card | 0.0209 | Enquiry |
| 13 | feature_7 | 0.0182 | Demographics |
| 14 | min_months_last_30_plus | 0.0181 | Account |
| 15 | feature_3 | 0.0167 | Demographics |

Most of the features suggested in the business case rank high here, for example `payment_history_avg_dpd_0_29_bucket`, `count_enquiry_recency_365`, `Ratio_currbalance_creditlimit` and `min_months_last_30_plus`. One exception is `utilisation_trend`. Dividing total-based utilisation by mean-based utilisation largely cancels out, so it added nothing beyond `Ratio_currbalance_creditlimit` and was not selected.

Full list: `feature_matrix_gain.csv`.

## Goal 3 — Model evaluation (Gini and rank ordering)

| Model | CV Gini | Train Gini | Test Gini | Rank breaks |
|---|---|---|---|---|
| **Logistic Regression (selected)** | 48.4 | 54.9 | **50.8** | 1 |
| Random Forest | 49.7 | 89.3 | 50.0 | 1 |
| XGBoost | 49.3 | 74.6 | 48.9 | 2 |
| LightGBM | 48.5 | 75.8 | 47.2 | 2 |
| Ensemble | – | 69.0 | 49.7 | 3 |
| *Benchmark* | | | *37.9* | |

Why Logistic Regression was chosen:

- It has the highest test Gini and the cleanest rank ordering.
- It shows the smallest gap between train and test Gini, so it generalises best.
- It is the easiest model to explain to credit-risk and audit teams.

The tree models overfit noticeably, scoring much higher on train than on test.

**Rank ordering (test set):**

| Decile | Bad rate | Benchmark | Cumulative bads captured |
|---|---|---|---|
| 10 | 18.89% | 11.71% | 37.4% |
| 9 | 8.89% | 8.05% | 55.1% |
| 8 | 4.89% | 6.00% | 64.8% |
| 7 | 4.22% | 3.96% | 73.1% |
| 6 | 4.00% | 4.25% | 81.1% |
| 5 | 3.33% | 3.66% | 87.7% |
| 4 | 2.44% | 2.05% | 92.5% |
| 3 | 2.67% | 2.05% | 97.8% |
| 2 | 0.67% | 2.05% | 99.1% |
| 1 | 0.44% | 1.32% | 100% |

- The top 20% of scores capture 55% of all bad customers.
- Max KS is 36.9.
- The only break is between deciles 3 and 4, which differ by one customer. The benchmark has a similar break between deciles 6 and 7.

## Cred score

The calibrated probability of default is turned into a score of 300–900:

- 600 points means 19:1 good:bad odds.
- Every 40 points doubles the odds.

| Risk band | Customers | Actual bad rate |
|---|---|---|
| Very High Risk (< 500) | 217 | 34.1% |
| High Risk (500–579) | 2,868 | 12.3% |
| Medium Risk (580–649) | 7,827 | 3.7% |
| Low Risk (650–719) | 3,832 | 0.9% |
| Very Low Risk (720+) | 256 | 0.8% |

**Suggested use:** customers below 580 go to limit review or tighter monitoring. Low and Very Low risk customers are candidates for limit increases and cross-sell.

*Note: these figures come from the sample data built to the client schema. Rerun on the real `project_banking` tables using `src/fetch_from_mysql.py`.*
