# EMIPredict AI - Intelligent Financial Risk Assessment Platform
### *Enterprise Dual-Target Machine Learning & Automated Loan Underwriting Engine*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Multi--Page%20App-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org)
[![Machine Learning](https://img.shields.io/badge/Models-XGBoost%20%7C%20RandomForest%20%7C%20Logistic%20%7C%20Ridge-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 Executive Overview
**EMIPredict AI** is an enterprise-grade financial risk assessment platform designed to automate loan underwriting for retail banks, credit unions, and FinTech digital lending platforms. 

Using 400,000 borrower profiles across five diverse lending scenarios, the platform executes a **Dual Machine Learning Architecture**:
1. **Multi-Class Classification:** Predicts loan eligibility status into `Eligible` (Prime), `High_Risk` (Subprime/Marginal), or `Not_Eligible` (High Risk).
2. **Continuous Regression:** Estimates the exact safe maximum monthly EMI repayment capacity (₹500 to ₹50,000) using 22 financial, demographic, and liquidity variables.

The system is fully integrated with **MLflow** for experiment tracking, parameter auditing, and model registry versioning, paired with an interactive **Streamlit Multi-Page Web Application** featuring real-time risk assessment, exploratory visual analytics, and administrative CRUD data management.

---

## 🚀 Key Business Impact
- **80% Reduction in Underwriting Turnaround:** Replaces 24-48 hour manual underwriting with sub-200 millisecond automated decisions.
- **Proactive NPA Reduction:** Identifies over-leveraged borrowers by analyzing free cash flow and Debt-to-Income (DTI) rather than raw salary alone.
- **Risk-Adjusted Pricing:** Suggests tailored mitigants (e.g., tenure extensions, co-applicants, or risk premiums) for `High_Risk` marginal applicants.

---

## 🏦 5 Lending Scenarios Supported
| Scenario | Typical Ticket Size | Repayment Tenure | Key Underwriting Focus |
| :--- | :--- | :--- | :--- |
| **E-commerce Shopping EMI** | ₹10,000 – ₹200,000 | 3 – 24 Months | Instant digital checkout, quick liquidity check |
| **Home Appliances EMI** | ₹20,000 – ₹300,000 | 6 – 36 Months | Household debt burden & disposable income |
| **Vehicle EMI** | ₹80,000 – ₹1,500,000 | 12 – 84 Months | Asset collateral, multi-year DTI stability |
| **Personal Loan EMI** | ₹50,000 – ₹1,000,000 | 12 – 60 Months | Unsecured credit risk, credit score threshold $\ge 680$ |
| **Education EMI** | ₹50,000 – ₹500,000 | 6 – 48 Months | Long-term family obligation buffering |

---

## 📊 Model Performance Leaderboard

### Classification Models (Target: `emi_eligibility` | Benchmark: > 90.0% Accuracy)
| Model Algorithm | Accuracy | Weighted F1-Score | Macro Precision | Macro Recall | ROC-AUC (OVR) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **98.95%** | **0.9895** | **0.9882** | **0.9891** | **0.9998** | 🏆 **Production Champion** |
| Random Forest Classifier | 97.32% | 0.9733 | 0.9710 | 0.9725 | 0.9982 | Challenger |
| Logistic Regression (Baseline) | 85.97% | 0.8674 | 0.8520 | 0.8610 | 0.9618 | Baseline |

### Regression Models (Target: `max_monthly_emi` | Benchmark: < ₹2,000.00 RMSE)
| Model Algorithm | RMSE (INR) | MAE (INR) | R² Score | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **₹273.92** | **₹175.98** | **0.9994** | **1.82%** | 🏆 **Production Champion** |
| Random Forest Regressor | ₹294.60 | ₹115.25 | 0.9993 | 1.15% | Challenger |
| Linear Ridge Regression | ₹2,641.46 | ₹1,846.83 | 0.9475 | 16.42% | Baseline |

---

## 🛠️ Feature Engineering & Statistical Significance
### 1. Engineered Financial Ratios
- **Debt-to-Income (DTI):** $\text{DTI} = \frac{\text{Current EMI Outflow}}{\text{Monthly Gross Salary}}$
- **Expense-to-Income (ETI):** $\text{ETI} = \frac{\text{Total Obligations}}{\text{Monthly Gross Salary}}$
- **Net Disposable Cash Flow:** $\text{FCF} = \text{Monthly Salary} - \text{Total Living & Debt Expenses}$
- **Requested Monthly Burden Ratio:** $\text{Burden Ratio} = \frac{\text{Requested Amount} / \text{Tenure}}{\text{Monthly Salary}}$
- **Financial Health Score (0-100):** Weighted multi-factor index combining CIBIL credit score, liquidity buffer, and debt leverage.

### 2. Statistical Testing Battery
- **Chi-Square Test ($\chi^2$):** Statistically significant associations ($p < 0.001$) between loan scenarios/employment types and applicant eligibility.
- **One-Way ANOVA:** Distinct group means confirmed ($p < 0.001$) across `Eligible`, `High_Risk`, and `Not_Eligible` for `salary`, `credit_score`, and `disposable_income`.
- **Pearson & Spearman Correlations:** Strong monotonic correlation ($r > 0.88$) between disposable cash flow and maximum safe EMI.

---

## 💻 Streamlit Multi-Page Web Application

The platform includes a modern multi-page Streamlit application located in `app/`:
1. **Home (`Home.py`):** Executive overview, platform KPIs, architecture data flow, and quick start guide.
2. **Predict EMI (`1_Predict_EMI.py`):** Real-time dual prediction interface with quick-load sample profiles (Prime, Marginal, High-Risk), instant eligibility badge, safe max EMI limit, and AI underwriter recommendations.
3. **EDA & Visual Insights (`2_EDA_Insights.py`):** Interactive exploratory dashboards addressing the 15 standard EDA questions with interactive boxplots, distributions, and correlation heatmaps.
4. **Model Evaluation (`3_Model_Evaluation.py`):** Live MLflow experiment leaderboard, metric comparison tables, confusion matrices, and feature importance charts.
5. **Manage Financial Data (`4_Manage_Data.py`):** Administrative portal providing full **CRUD** operations (Create, Read, Update, Delete) on applicant records.

---

## 📁 Repository Structure
```
1st project/
│
├── data/
│   ├── raw/                           <- Raw dataset directory (emi_prediction_dataset.csv)
│   ├── processed/                     <- Preprocessed train/val/test splits
│   └── admin_financial_records.csv    <- CRUD administrative database
│
├── src/                               <- Reusable modular Python package
│   ├── __init__.py
│   ├── config.py                      <- Configuration, paths, feature lists, thresholds
│   ├── data_loader.py                 <- Ingestion, synthetic fallback, train-test splits
│   ├── preprocessing.py               <- Scikit-learn ColumnTransformer pipeline
│   ├── feature_engineering.py         <- Domain financial ratios & health scores
│   ├── statistical_tests.py           <- Chi-Square, ANOVA, Pearson/Spearman battery
│   ├── classification_models.py      <- 3 classification models (Logistic, RF, XGBoost)
│   ├── regression_models.py           <- 3 regression models (Ridge, RF, XGBoost)
│   ├── mlflow_utils.py                <- MLflow tracking, artifacts, and registry
│   ├── predict.py                     <- Real-time dual inference engine
│   └── pipeline.py                    <- Master end-to-end training pipeline
│
├── models/                            <- Serialized production models & artifacts
│   ├── best_classifier.joblib         <- Champion XGBoost Classifier
│   ├── best_regressor.joblib          <- Champion XGBoost Regressor
│   ├── preprocessor.joblib            <- Fitted ColumnTransformer
│   ├── metrics_summary.json           <- Summary evaluation metrics
│   ├── XGBoost_confusion_matrix.png   <- Confusion matrix visualization
│   └── XGBoost_feature_importance.png <- Top feature importance chart
│
├── app/                               <- Streamlit Multi-Page Web Application
│   ├── Home.py                        <- Main landing page & dashboard
│   └── pages/
│       ├── 1_Predict_EMI.py           <- Real-time dual prediction engine
│       ├── 2_EDA_Insights.py          <- Interactive EDA dashboards
│       ├── 3_Model_Evaluation.py      <- MLflow comparison leaderboard
│       └── 4_Manage_Data.py           <- Full CRUD administrative portal
│
├── docs/                              <- Comprehensive capstone documentation
│   ├── EDA_Report.md                  <- 15-question structured EDA report
│   ├── Capstone_Defense_Script.md     <- 30-minute evaluation defense script
│   ├── Mock_Interview_QA.md           <- Top 25 technical & FinTech interview Q&As
│   ├── chi_square_tests.csv           <- Chi-Square test statistical results
│   ├── anova_tests.csv                <- ANOVA test statistical results
│   └── correlation_tests.csv          <- Correlation matrix analysis
│
├── requirements.txt                   <- Pinned production dependencies
└── README.md                          <- Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/EMIPredict-AI.git
cd EMIPredict-AI

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Training Pipeline
```bash
# Sets PYTHONPATH and trains all 6 models with MLflow tracking
python -m src.pipeline
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app/Home.py
```
Open your browser at `http://localhost:8501`.

### 4. Launch the MLflow Experiment Tracking UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser at `http://localhost:5000` to interactively compare runs, metrics, and registered models.

---

## ☁️ Streamlit Cloud Deployment Instructions
1. Push this entire repository to your GitHub account.
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Click **"New App"** and select your GitHub repository and branch.
4. Set the **Main file path** to `app/Home.py`.
5. Click **"Deploy"**. Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and launch your live application.

---

## 🎓 Capstone Live Evaluation Preparation
For the 30-minute live project review:
- Refer to **[`docs/Capstone_Defense_Script.md`](docs/Capstone_Defense_Script.md)** for the exact timing breakdown (3-min intro, 12-min technical explanation, 10-min Q&A).
- Refer to **[`docs/Mock_Interview_QA.md`](docs/Mock_Interview_QA.md)** for prepared answers to evaluator technical questions.
- Refer to **[`docs/EDA_Report.md`](docs/EDA_Report.md)** for the detailed 15-question EDA defense.
