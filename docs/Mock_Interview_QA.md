# Comprehensive Mock Interview Q&A
## EMIPredict AI - FinTech Risk Assessment Platform
*Prepared for the 10-Minute Technical Mock Interview (10 Marks in Live Evaluation)*

---

### Category 1: Python & Modular Software Architecture

#### Q1: Why did you structure the project as a modular package rather than writing everything in a single Jupyter Notebook?
**Model Answer:**
> *"Notebooks are fantastic for scratch exploration and visual inspection, but they suffer from hidden cell-execution state, poor version control diffs, and lack of testability. In production FinTech systems, reliability is paramount. I isolated data ingestion (`data_loader.py`), domain transformations (`feature_engineering.py`), preprocessor pipelines (`preprocessing.py`), hypothesis tests (`statistical_tests.py`), and model suites (`classification_models.py`, `regression_models.py`) into decoupled modules. This adheres to the Single Responsibility Principle (SRP), enables automated unit testing, guarantees reproducibility, and allows our Streamlit frontend to directly import inference functions without notebook dependencies."*

#### Q2: How did you handle potential data leakage in your preprocessing and scaling steps?
**Model Answer:**
> *"Data leakage occurs when information from outside the training dataset is inadvertently used to create the model. In our pipeline, we first split the dataset into stratified Train, Validation, and Test sets using `train_test_split`. We then fit the `ColumnTransformer` (which computes median values for imputation and mean/standard deviation for `StandardScaler`) **strictly on the training split**. The validation and test sets—as well as real-time applicant inputs in Streamlit—are only processed using the `.transform()` method of the fitted preprocessor pipeline, guaranteeing zero contamination."*

---

### Category 2: Data Preprocessing & Feature Engineering

#### Q3: What domain-specific financial ratios did you engineer, and why are they superior to raw variables?
**Model Answer:**
> *"Raw income alone is a misleading indicator of credit capacity; a high-earning individual with heavy debt obligations can have negative free cash flow. We engineered:
> 1. **Debt-to-Income (DTI):** `current_emi / monthly_salary`, measuring existing leverage.
> 2. **Expense-to-Income (ETI):** Total living expenses relative to income.
> 3. **Net Disposable Cash Flow:** Gross income minus rent, school fees, utilities, and current EMIs.
> 4. **Requested Monthly Burden Ratio:** `(requested_amount / tenure) / salary`.
> 5. **Composite Financial Health Score (0-100):** Combining credit score, liquidity buffer, and debt leverage.
> These ratios directly capture debt servicing capability and improved our classification F1-score from 0.86 in baseline models to 0.989 in tree models."*

#### Q4: How did you detect and handle outliers in financial features?
**Model Answer:**
> *"Financial variables (such as monthly income or requested loan amounts) typically exhibit heavy right skewness. Completely dropping extreme values would discard legitimate high-net-worth customers. We used domain-informed Interquartile Range (IQR) capping (winsorization) at the 1st and 99th percentiles with a $1.5 \times \text{IQR}$ threshold, while enforcing non-negativity constraints ($x \ge 0$) on monetary amounts."*

---

### Category 3: Statistical Hypothesis Testing

#### Q5: Why did you perform Chi-Square tests and ANOVA tests prior to model training?
**Model Answer:**
> *"To ensure that feature inclusion was justified through rigorous statistical inference rather than arbitrary selection:
> - **Chi-Square Test of Independence ($\chi^2$):** Assessed whether categorical attributes (`education`, `employment_type`, `emi_scenario`) had a statistically significant association with the multi-class target `emi_eligibility` ($p < 0.05$). All features showed $p < 0.001$, confirming strong predictive validity.
> - **One-Way ANOVA ($F$-Test):** Tested whether continuous features (`salary`, `credit_score`, `disposable_income`) showed distinct population means across `Eligible`, `High_Risk`, and `Not_Eligible` groups ($p < 0.001$), confirming their capability to separate borrower classes."*

---

### Category 4: Machine Learning (Classification & Regression)

#### Q6: Why did you choose XGBoost over Random Forest and Logistic Regression as your production champion?
**Model Answer:**
> *"While Random Forest performed well (97.32% accuracy), **XGBoost achieved the superior benchmark of 98.95% accuracy and 0.9895 F1-Score for classification, and an RMSE of ₹273.92 (R² = 0.9994) for regression**. XGBoost achieves this by:
> 1. Sequentially minimizing residuals using second-order Taylor approximations (gradient and Hessian).
> 2. Built-in $L_1$ (Lasso) and $L_2$ (Ridge) leaf weight regularization preventing overfitting on complex financial interactions.
> 3. Native handling of non-linear non-monotonic interactions between credit score and debt-to-income ratios."*

#### Q7: Which evaluation metrics did you prioritize for this dual-target FinTech problem, and why?
**Model Answer:**
> *"For Classification, Accuracy alone is insufficient due to class imbalance; a false negative (classifying a defaulting borrower as 'Eligible') costs a bank significantly more than a false positive. We prioritized **Macro and Weighted F1-Score** alongside **Multi-class One-vs-Rest ROC-AUC**, ensuring high sensitivity across the `High_Risk` subprime segment.
> For Regression, we prioritized **RMSE (Root Mean Squared Error)** because it heavily penalizes large dollar prediction errors in loan sizing, supplemented by **MAE** and **R²** to measure total explained variance."*

---

### Category 5: MLflow Experiment Tracking & Governance

#### Q8: What did you track with MLflow, and how is it used in production?
**Model Answer:**
> *"We utilized an SQLite-backed MLflow tracking server (`sqlite:///mlflow.db`). For every model variant across classification and regression, we programmatically logged:
> 1. **Hyperparameters:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `class_weight`.
> 2. **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, R².
> 3. **Visual Artifacts:** Confusion matrix heatmaps, ROC curves, and Tree Feature Importance plots.
> 4. **Model Serialization:** Logged model packages and registered the champion models into the MLflow Model Registry for version-controlled deployment in Streamlit."*

---

### Category 6: Streamlit Application & Production Deployment

#### Q9: How does the Streamlit web application fulfill business requirements?
**Model Answer:**
> *"The multi-page application delivers 4 core business capabilities:
> 1. **Home Dashboard:** High-level platform KPIs, executive problem statement, and scenario architecture.
> 2. **Real-Time Dual Prediction:** Interactive forms and quick-load presets for live loan underwriting, returning immediate eligibility badges, max safe EMI limits, risk flags, and underwriter recommendations.
> 3. **Interactive EDA Insights:** Dynamic visualizations answering the 15 standard EDA questions.
> 4. **Model Evaluation:** Direct connection to the MLflow database displaying an auditable leaderboard.
> 5. **Data Management:** Complete CRUD interface allowing administrators to create, search, update, and delete applicant records."*

#### Q10: How do you deploy this application to Streamlit Cloud?
**Model Answer:**
> *"The repository is structured with a root `requirements.txt` containing pinned production packages (`streamlit`, `mlflow`, `xgboost`, `scikit-learn`, `joblib`). Once pushed to GitHub, Streamlit Community Cloud connects directly to the repository, specifies `app/Home.py` as the entrypoint, auto-installs dependencies from `requirements.txt`, and serves the live application via a secure, public HTTPS URL."*
