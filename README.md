# PRCL-0015 · Bank GoodCredit — Credit Score (Cred Score) Prediction

**Client:** Bank GoodCredit **Category:** Banking – Risk **Project ref:** PM-PR-0015

Bank GoodCredit wants to predict a *cred score* for its current credit-card customers. The score measures creditworthiness and helps the bank reduce credit-default risk.

- **Target:** `Bad_label` → 0 = good credit history, 1 = bad credit history (30+ DPD bucket)
- **Benchmark:** Gini **37.9** (66-feature ensemble)

## Results

| Metric | Benchmark | This project |
|---|---|---|
| Test Gini | 37.9 | **50.8** (Logistic Regression) |
| Bad rate, top decile | 11.7% | **18.9%** |
| Bad rate, bottom decile | 1.3% | **0.4%** |
| Max KS | – | 36.9 |
| Features used | 66 | 50 |

All five models beat the benchmark:

| Model | CV Gini | Train Gini | Test Gini | Rank breaks |
|---|---|---|---|---|
| **Logistic Regression** | 48.4 | 54.9 | **50.8** | 1 |
| Random Forest | 49.7 | 89.3 | 50.0 | 1 |
| XGBoost | 49.3 | 74.6 | 48.9 | 2 |
| LightGBM | 48.5 | 75.8 | 47.2 | 2 |
| Ensemble (XGB + LGBM + LR) | – | 69.0 | 49.7 | 3 |

**Rank ordering of the best model (test set, 4,500 customers):**

| Decile | Customers | Bads | Bad rate | Benchmark |
|---|---|---|---|---|
| 10 | 450 | 85 | 18.89% | 11.71% |
| 9 | 450 | 40 | 8.89% | 8.05% |
| 8 | 450 | 22 | 4.89% | 6.00% |
| 7 | 450 | 19 | 4.22% | 3.96% |
| 6 | 450 | 18 | 4.00% | 4.25% |
| 5 | 450 | 15 | 3.33% | 3.66% |
| 4 | 450 | 11 | 2.44% | 2.05% |
| 3 | 450 | 12 | 2.67% | 2.05% |
| 2 | 450 | 3 | 0.67% | 2.05% |
| 1 | 450 | 2 | 0.44% | 1.32% |

**Top features by gain:** `mean_dpd`, `n_accounts_ever_30p`, `pct_accounts_ever_30p`, `payment_history_avg_dpd_0_29_bucket`, `payment_history_total_dpd_0_29`, `count_enquiry_recency_365`, `Ratio_currbalance_creditlimit`. Full list: `reports/feature_matrix_gain.csv`.

> **About the data.** The client database (`project_banking` on MySQL) was not reachable when this project was built. `data/raw/` therefore holds **sample data generated to the exact schema** of the three tables (same columns, date format and payment-history codes, ~5% bad rate). The numbers above come from that sample data. To use the real tables, follow *Using the real database* below. The whole pipeline then reruns unchanged.

## Project structure

```
PRCL-0015_Bank_GoodCredit/
├── data/
│   ├── raw/
│   │   ├── customer_account.csv        # Cust_Account      (96,361 rows × 21)
│   │   ├── customer_enquiry.csv        # Cust_Enquiry      (116,326 rows × 6)
│   │   └── customer_demographics.csv   # Cust_Demographics (15,000 rows × 83, incl. Bad_label)
│   └── processed/
│       └── feature_matrix.csv          # 1 row per customer, 121 engineered features
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # Goal 1: insights + decisions taken
│   ├── 02_Feature_Engineering.ipynb    # building and checking features
│   └── 03_Model_Building_Evaluation.ipynb  # Goals 2 & 3: feature gain, Gini, rank ordering
├── src/
│   ├── generate_data.py                # builds the sample datasets
│   ├── fetch_from_mysql.py             # pulls the real tables from MySQL
│   ├── feature_engineering.py          # raw tables → customer feature matrix
│   ├── model_utils.py                  # calibration / ensemble wrappers
│   ├── train.py                        # selection, training, evaluation, charts
│   └── predict.py                      # cred score (300–900) + risk band per customer
├── models/
│   ├── credit_score_model.pkl          # best model (calibrated) + feature list
│   └── selected_features.json
├── reports/
│   ├── Project_Findings.md             # written summary of all three goals
│   ├── model_comparison.csv
│   ├── rank_ordering.csv
│   ├── feature_matrix_gain.csv
│   ├── customer_cred_scores.csv
│   ├── model_summary.json
│   └── figures/                        # ROC, rank ordering, gains/KS, feature gain, model comparison
├── docs/Business_Case_PRCL-0015.pdf
├── requirements.txt
└── .env.example
```

## How to run

```bash
pip install -r requirements.txt

python src/generate_data.py         # optional: regenerate the sample data
python src/feature_engineering.py   # -> data/processed/feature_matrix.csv
python src/train.py                 # -> models/ and reports/
python src/predict.py               # -> reports/customer_cred_scores.csv
python src/predict.py --customer 1007
```

Or open the notebooks in order (`jupyter notebook notebooks/`).

## Using the real database

```bash
cp .env.example .env      # fill in host, user and password from the business case PDF
python src/fetch_from_mysql.py
python src/feature_engineering.py
python src/train.py
```

`fetch_from_mysql.py` overwrites the three CSVs in `data/raw/` with the tables `Cust_Account`, `Cust_Enquiry` and `Cust_Demographics`. Keep `.env` out of version control.

## Method in brief

1. **Exploration:** target imbalance, missing values, payment-history coding, bad rate against delinquency, utilisation and enquiries (Notebook 1).
2. **Feature engineering:**
   - **Account table:** payment-history strings are parsed into monthly DPD. From these come 0-29 / 30+ / 60+ / 90+ counts, max and mean DPD, and months since the last 30+ DPD. Further features cover utilisation, balance ratios, account age and last-payment gaps.
   - **Enquiry table:** counts over 30/90/180/365-day windows, amounts and purposes.
   - **Demographics:** columns with more than 70% missing, constant columns and ID-like columns are dropped. Remaining gaps are filled with the median. Text columns are frequency-encoded.
3. **Feature selection (train data only):** drop features with |corr| > 0.95. Then run a shadow-feature test: shuffled copies of real columns act as pure noise, and a feature is kept only if its XGBoost gain beats the best noise column.
4. **Models:** Logistic Regression, Random Forest, XGBoost, LightGBM and an Ensemble. Class weights handle the 5% bad rate. Evaluation uses 5-fold CV and a stratified 70/30 split.
5. **Choice and calibration:** the chosen model has the fewest rank-order breaks and then the highest test Gini. Platt scaling makes the predicted default rate match the real one (5.1% vs 5.0%) without changing the ranking.
6. **Cred score:** 600 points at 19:1 good:bad odds, with 40 points to double the odds. Scores are capped to 300–900 and grouped into five risk bands.
