# SaaS Churn Intelligence - End-to-end ML Project

## Overview
This project builds an end-to-end churn prediction system for a fictional SaaS platform (**RavenStack**) using realistic, multi-table product telemetry.  
The goal is to predict whether an account will churn within the next **30 days**, using **weekly behavioral snapshots** and business-aligned evaluation metrics.

The project emphasizes:
- Leakage-safe time-based modeling
- Interpretable feature engineering
- Business-driven evaluation (ranked outreach)
- Clear bias–variance reasoning

No dashboards are used; all insights are delivered via reproducible Python notebooks.

---

## Dataset
The dataset is fully synthetic and designed to simulate a real SaaS product.
Link: https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset

**Tables:**
- accounts — customer metadata
- subscriptions — plan lifecycle and revenue
- feature_usage — daily product usage
- support_tickets — support activity
- churn_events — churn dates and reasons

**Key properties:**
- Referential integrity across tables
- Temporal realism (signup → usage → churn)
- Built-in edge cases (trials, upgrades, reactivations)

---

## Problem Definition
**Objective:**  
Predict whether an account will churn in the next **30 days**, using only information available up to the prediction date.

**Prediction setup:**
- Weekly snapshots (every Monday)
- One row = one account at one point in time
- Forward-looking churn labels (no leakage)

---

## Project Structure

``
saas-churn-intelligence/
│
├── data/
│ ├── raw/ # Original CSV files
│ └── processed/ # Snapshot & feature datasets
│
├── notebooks/
│ ├── 01_eda_data_audit.ipynb
│ ├── 02_data_processing_snapshot.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_baseline_overfit_underfit.ipynb
│ ├── 05_hyperparameter_tuning.ipynb
│ └── 06_model_evaluation.ipynb
│
├── models/
│ ├── best_lr_pipeline.joblib
│ ├── best_lr_model_meta.json
│ ├── best_hgb_pipeline.joblib
│ └── best_hgb_model_meta.json
│
├── reports/
│ └── model_evaluation_results.json
│
├── src/ # (Reserved for productionization)
├── requirements.txt
└── README.md
``

---

## Methodology

### 1. Data Audit & Validation
- Row counts and schema validation
- Referential integrity checks
- Missing value analysis
- Date range verification

### 2. Snapshot Construction
- Weekly account snapshots
- Excludes post-churn data
- Labels based on churn within next 30 days

### 3. Feature Engineering (Interpretable)
- Usage behavior (levels, trends, normalization)
- Support friction signals
- Subscription stability indicators
- Account maturity features

### 4. Baseline Modeling
- Logistic Regression
- Time-based train/validation/test split
- Demonstrates intentional underfitting

### 5. Model Improvement
- Class imbalance handling
- Regularization tuning
- Gradient Boosting comparison

### 6. Evaluation
- ROC-AUC and PR-AUC
- Recall & Precision at Top-20% outreach
- Confusion matrices
- Business-oriented interpretation

---

## Key Results

**Champion model:** Logistic Regression (balanced, tuned)

**Test-set performance (Top-20% outreach):**
- ~42% of churners identified
- ~31 churners per 100 contacted accounts
- Strong generalization to unseen time periods

The tree-based model did not outperform the linear model due to the deliberately aggregated and interpretable feature design.

---

## Business Interpretation
> If the retention team contacts the top 20% highest-risk accounts each week, the model can identify a substantial portion of upcoming churn while maintaining practical outreach efficiency.

---

## Limitations & Future Work
- Add raw, high-granularity feature interactions
- Incorporate textual churn feedback
- Extend to uplift modeling (who to contact vs who will churn)
- Production deployment with scheduled retraining

---

## Tech Stack
- Python, pandas, NumPy
- scikit-learn
- Jupyter
- Joblib (model artifacts)

---
