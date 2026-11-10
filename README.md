# Telecom Customer Churn Prediction & Retention Intelligence System

An interactive machine learning dashboard for estimating customer churn risk, exploring model explanations, prioritizing outreach, and scoring customer datasets in bulk.

> **Project status:** Functional local prototype; deployment pending; notebook leakage review completed. This is an educational demonstration, not a validated production decision system.

## Overview

Customer churn occurs when subscribers stop using a service. This project uses customer account, service, and billing features to estimate the probability of churn and presents the results in a Streamlit dashboard. It combines predictive analytics with interpretable model outputs and transparent, rule-based retention ideas.

## Features

| Dashboard page | What it does |
| --- | --- |
| **Overview** | Shows customer counts, high-risk share, mean predicted churn risk, risk distribution, contract-level patterns, and a searchable customer table. |
| **Churn Prediction** | Accepts customer feature inputs and returns a predicted churn probability, risk category, and suggested follow-up actions. |
| **Explainable AI** | Displays logistic regression coefficients and individual SHAP waterfall explanations. |
| **Retention Intelligence** | Filters customers by risk segment, ranks a customer queue, and presents rule-based outreach ideas. |
| **Batch Predictions** | Scores an uploaded CSV, summarizes risk distribution, and exports scored predictions. |

### Risk segments

- **Low:** predicted probability below 30%
- **Medium:** 30% to below 60%
- **High:** 60% and above

These are application-defined operational thresholds, not independently validated business cutoffs.

## Tech stack

- **Python**, **Pandas**, **NumPy** — data handling and preprocessing
- **scikit-learn 1.6.1** — trained classification pipeline
- **joblib** — saved model loading
- **Streamlit** — interactive application
- **Plotly** — dashboard charts and risk gauges
- **SHAP**, **Matplotlib** — individual prediction explanations
- **Google Colab / Jupyter Notebook** — experimentation and training

## Dataset and evaluation scope

**Source:** IBM Telco Customer Churn sample dataset (also distributed on Kaggle). The training notebook retrieves the CSV from the [IBM GitHub dataset repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv), with a manual-upload fallback. Add the specific Kaggle listing if used and review applicable redistribution terms before publishing data files.

| Split | Customers |
| --- | ---: |
| Total dataset | 7,043 |
| Training set | 5,634 (80%) |
| Held-out test set | 1,409 (20%) |

The bundled `customer_risk_predictions.csv` contains **1,409 test-set records**, including actual outcomes and predicted probabilities. The Overview and Retention Intelligence pages describe this test-set portfolio. They should **not** be interpreted as measurements across all 7,043 customers or as evidence of live business outcomes.

**Verified consistency checks:**

- Average test-set predicted churn probability in Colab: **41.56%**; dashboard displays **41.6%** after rounding.
- Five individually compared prediction probabilities matched between Colab and the locally loaded model to six decimal places.

These checks establish consistency for the tested examples; they do not replace a full prediction-by-prediction audit, calibration assessment, or external validation. The notebook was reviewed for common train/test leakage paths; none was apparent in the training pipeline.

### Model evaluation

The app loads evaluation information from `model_report.json`, accessible under **Explainable AI → Model evaluation report**. The following metrics are from the held-out test set:

| Metric | Test-set value |
| --- | --- |
| Accuracy | **73.95%** |
| Precision | **50.60%** |
| Recall | **78.34%** |
| F1 score | **61.49%** |
| ROC-AUC | **84.04%** |
| Average precision | **62.66%** |

The selected model is **Logistic Regression**, with tuned regularization parameter `model__C = 18.73817422860385`. At the evaluation threshold used to produce the report, recall is higher than precision: the model catches many actual churners but also flags some non-churners. The notebook calculates these metrics using a **0.50 classification threshold** (`test_pred = (test_proba >= 0.5)`); the dashboard's **0.30 and 0.60 risk-segment thresholds** serve a different purpose. The model uses `class_weight="balanced"`, and predicted probabilities have not been shown to be calibrated.

## How the system works

1. Remove `Churn` (target) and `customerID` (identifier) from input features; make an 80/20 stratified train/test split.
2. Compare Logistic Regression and Random Forest with five-fold stratified cross-validation on training data; tune the selected model with `RandomizedSearchCV`.
3. Fit imputation, scaling, and one-hot encoding **inside** the training pipeline to avoid fitting transformations on the test set.
4. Save the fitted pipeline as `churn_pipeline.joblib` and the model metadata as `model_report.json`.
5. Load the model and customer test-set CSV in Streamlit.
6. Use `predict_proba` to estimate churn probabilities.
7. Group scores into Low, Medium, and High risk segments.
8. Explore coefficients and SHAP explanations; review heuristic retention suggestions.
9. Upload additional CSVs containing the required model input columns to score them in bulk.

**Interpretation note:** Logistic regression coefficients and SHAP values describe model behavior, not causal drivers. SHAP values in this app are expressed in **log-odds**, not percentage-point changes in churn probability. Retention suggestions are rules based on customer attributes and have not been experimentally validated.

## Project structure

```text
telecom-churn-intelligence/
├── app.py
├── churn_pipeline.joblib
├── model_report.json
├── customer_risk_predictions.csv   # Include only after checking source permissions
├── requirements.txt
├── Telecom_Churn_Prediction_Complete_Colab.ipynb
├── README.md
└── .gitignore
```

Local virtual environments, backups, and debugging scripts should remain outside the Git repository.

## Run locally

Use **Python 3.13** and the dependency versions in `requirements.txt`. The trained pipeline was saved with **scikit-learn 1.6.1**; using an incompatible version may prevent it from loading.

```powershell
# From the project directory (Windows PowerShell)
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe -m streamlit run app.py
```

Ensure `churn_pipeline.joblib`, `model_report.json`, and (for portfolio views) `customer_risk_predictions.csv` are in the same directory as `app.py`.

## Dashboard screenshots

Add screenshots after creating an `assets/` folder:

- Overview — `assets/overview.png`
- Churn Prediction — `assets/churn-prediction.png`
- Explainable AI — `assets/explainable-ai.png`
- Retention Intelligence — `assets/retention-intelligence.png`
- Batch Predictions — `assets/batch-predictions.png`

## Live demo

**Streamlit app:** _Add deployment URL after publishing._

## Limitations and responsible use

- The bundled portfolio is a **test-set sample**, not a live telecom customer base.
- Predicted churn probabilities are model estimates; they are not guaranteed outcomes.
- Risk thresholds and outreach suggestions are heuristic and require business validation.
- The model may reflect biases or limitations in the original dataset; assess fairness, calibration, and data leakage before operational use.
- Do not upload or publish confidential customer information. Check source permissions before committing the bundled CSV.
- Do not load untrusted `.joblib` files: joblib/pickle deserialization can execute arbitrary code.

## Next steps

- Notebook code review completed: the target and customer identifier are excluded; preprocessing is fitted within the cross-validation/training pipeline. Consider a deeper audit and independent validation before business use.
- Add dashboard screenshots and exact dataset citation.
- Review CSV redistribution terms.
- Publish the repository and deploy on Streamlit Community Cloud.

## Author

Data Science portfolio project. Add your preferred public name and LinkedIn/GitHub profile links before publishing.
