# Capstone Project Presentation & Live Defense Script
## EMIPredict AI - Intelligent Financial Risk Assessment Platform
*Calibrated for the Mentor-Instructed 17 to 20-Minute Master Video Presentation (Target: 18 Minutes 30 Seconds).*

---

## 🎥 Master Video Presentation Schedule (17 to 20 Minutes)

| Part | Section | Time Allocation | Cumulative | Screen Action / Visual Cue | Core Speaking Focus |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | **Why & Problem Statement** | **2.5 Min** | 00:00 – 02:30 | Webcam / Title Slide | Retail credit boom, 2 Fatal Flaws (Salary Multiple Fallacy, Multi-line Credit Blindness), & Dual-Task solution. |
| **2** | **VS Code File-by-File Tour** | **3.0 Min** | 02:30 – 05:30 | VS Code File Tree | Highlight all 10 core files (`config`, `data_loader`, `feature_engineering`, `preprocessing`, `statistical_tests`, `classification`, `regression`, `mlflow`, `pipeline`, `predict`, `app/`). |
| **3** | **5 Core Code Logics Breakdown** | **3.5 Min** | 05:30 – 09:00 | VS Code Code Lines | Line-by-line breakdown: Split-Before-Fit leakage, Outlier capper, DTI & Health score math, XGBoost Taylor params, Dual inference. |
| **4** | **Scientific Hypothesis Testing** | **2.0 Min** | 09:00 – 11:00 | Test CSVs / EDA Page | Chi-Square ($p < 0.001$), One-Way ANOVA ($F > 450$), Pearson & Spearman correlations ($r > 0.88$). |
| **5** | **Model Tournament & Results** | **3.0 Min** | 11:00 – 14:00 | Confusion Matrix & Parity Plot | 98.95% accuracy, ₹273.92 RMSE, 0.00% Eligible vs Ineligible error, homoscedastic residuals, Gini feature gain. |
| **6** | **Live Streamlit App Demo** | **2.75 Min** | 14:00 – 16:45 | Browser (Streamlit App) | 3 Personas (Reeva Shah, Priya Patel, Vikram Singh), Plotly gauges, Waterfall cash flow, and What-If sandbox. |
| **7** | **Future Expansion Roadmap & Closing** | **1.75 Min** | 16:45 – 18:30 | Architecture Diagram / Webcam | 4 Pillars: Account Aggregator (AA), Explainable AI (SHAP/LIME), Kafka/FastAPI streaming, Evidently AI MLOps drift. |

---

### Part 1: Why Am I Doing This Project & The Problem Statement (00:00 – 02:30 | 2.5 Min)
> **[SCREEN ACTION: Start with your webcam on screen or project title slide in fullscreen. Speak with energy, authority, and enterprise presence.]**
>
> *"Good day, respected evaluators and mentor. My name is Devansh. Today, I am proud to present **EMIPredict AI**, an enterprise-grade financial risk assessment and dual-target loan underwriting intelligence platform built for modern retail banks and FinTech lenders.
>
> **Why am I doing this project?**
> Across the globe and especially in emerging economies like India, retail consumer credit is expanding at an unprecedented rate of over **25% compound annual growth**. Millions of individuals are applying for consumer credit across e-commerce shopping, home appliances, two-wheelers, education, and personal loans. However, banking institutions are experiencing a severe surge in **early defaults and Non-Performing Assets (NPAs)**.
>
> When I investigated why this occurs, I discovered that traditional banking loan underwriting relies on two catastrophic flaws:
> 1. **The Gross Salary Multiple Fallacy:** Traditional banks evaluate an applicant based on gross income—for instance, *'You earn ₹80,000 per month, so we can grant you a ₹20,000 EMI'*. But they completely ignore fixed non-discretionary living overheads—such as house rent, children's school fees, groceries, and utilities. If an applicant earns ₹80,000 but pays ₹75,000 in living costs, their true free cash flow is only ₹5,000. Lending them a ₹20,000 EMI guarantees immediate default.
> 2. **Multi-Line Credit Blindness:** Modern digital consumers hold multiple simultaneous credit lines—a BNPL e-commerce EMI, a smartphone loan, and existing personal credit cards. Traditional rules evaluate each new loan in a silo, blind to how combined debt pushes the borrower's total **Debt-to-Income (DTI) ratio beyond the 40% danger cliff**.
>
> **How does EMIPredict AI solve this?**
> We solve this crisis by engineering a **Dual-Task Machine Learning Architecture** evaluated across **400,000 borrower records** spanning 5 diverse lending scenarios. Rather than a crude one-size-fits-all threshold, our platform computes true unencumbered disposable cash flow and solves two simultaneous questions in under 200 milliseconds:
> - **First**, multi-class risk classification—determining whether the applicant is *Eligible*, *High Risk*, or *Not Eligible* with **98.95% accuracy**.
> - **Second**, continuous affordability regression—calculating the exact, safe maximum monthly EMI capacity in INR with an RMSE of just **₹273.92**.
>
> Let me now transition to VS Code and walk you through the codebase file-by-file."*

---

### Part 2: Codebase Tour in VS Code — Showing Files & Module Architecture (02:30 – 05:30 | 3.0 Min)
> **[SCREEN ACTION: Switch screen sharing to VS Code. Open the file explorer sidebar on the left showing `src/` and `app/`. Click into each file as you introduce it.]**
>
> *"Now, let us examine the engineering structure of the project inside VS Code. As you can see in the explorer tree, the codebase is architected into modular, production-ready Python packages under `src/`, complemented by our interactive multi-page Streamlit application under `app/`. Let me show you what each specific file does:
>
> 1. **`src/config.py`:** [Click into `config.py`, highlight lines 24–64 and lines 69–96]
>    This file is our centralized configuration nerve center. It defines absolute project paths, the schemas for our 17 raw numerical features and 8 categorical features, the target definitions, our class mapping dictionary (Eligible = 0, High_Risk = 1, Not_Eligible = 2), and our strict capstone benchmarks—requiring over 90% classification accuracy and an RMSE under ₹2,000.
>
> 2. **`src/data_loader.py`:** [Click into `data_loader.py`, highlight `split_data()` lines 175–200]
>    This module handles data ingestion and validation. If the full 400K dataset is downloaded, it stages it into `data/raw/`. Furthermore, it contains our `split_data()` function, which performs stratified Train/Validation/Test splitting—70% for training, 10% for validation, and 20% for final test evaluation—guaranteeing zero data leakage.
>
> 3. **`src/feature_engineering.py`:** [Click into `feature_engineering.py`, highlight lines 40–77]
>    This is the heart of our domain intelligence. Rather than feeding raw unnormalized numbers into an algorithm, this module calculates real-world financial ratios: unencumbered disposable cash flow, Debt-to-Income (DTI), Expense-to-Income (ETI), Requested Monthly Burden, and our calibrated 0–100 Composite Financial Health Score.
>
> 4. **`src/preprocessing.py`:** [Click into `preprocessing.py`, highlight lines 34–66 and lines 68–95]
>    Here we construct our production Scikit-Learn `ColumnTransformer`. It includes our custom `OutlierCapper`, which applies IQR Winsorization with strict non-negativity floors, alongside median imputers, `StandardScaler` for continuous variables, and `OneHotEncoder` with unknown category handling for categoricals.
>
> 5. **`src/statistical_tests.py`:** [Click into `statistical_tests.py`, highlight lines 19–64]
>    This module implements our hypothesis testing battery using SciPy. It runs Chi-Square tests of independence for categorical variables, One-Way ANOVA tests across eligibility risk tiers, and Pearson/Spearman correlations against safe EMI capacity.
>
> 6. **`src/classification_models.py` & `src/regression_models.py`:** [Click into both files]
>    These two modules define and benchmark our candidate model suites: Logistic Regression, Random Forest Classifier, and XGBoost Classifier for multi-class risk; and Linear Ridge, Random Forest Regressor, and XGBoost Regressor for continuous EMI capacity.
>
> 7. **`src/mlflow_utils.py`:** [Click into `mlflow_utils.py`, highlight lines 20–70]
>    This file encapsulates enterprise MLOps. It connects to an SQLite database tracking URI (`sqlite:///mlflow.db`), logging parameters, loss metrics, confusion matrix heatmaps, parity plots, and serialized model binaries.
>
> 8. **`src/pipeline.py` & `src/predict.py`:** [Click into `pipeline.py` and `predict.py`]
>    `pipeline.py` is the master training orchestrator that runs the entire end-to-end workflow from raw data to saved models in `models/`. `predict.py` provides our real-time `EMIPredictor` class, which loads the serialized models and executes sub-200ms dual inference with automated underwriting risk explanations.
>
> 9. **`app/` Directory:** [Expand `app/` folder in explorer]
>    Finally, our `app/` folder contains our production Streamlit suite: `Home.py`, `1_Predict_EMI.py` for the Underwriting Studio, `2_EDA_Insights.py`, `3_Model_Evaluation.py`, and `4_Manage_Data.py` for administrative CRUD operations."*

---

### Part 3: 5 Critical Code Logics Deep Dive — Line-by-Line Breakdown (05:30 – 09:00 | 3.5 Min)
> **[SCREEN ACTION: In VS Code, navigate directly to each file and highlight the exact line numbers specified below so the evaluator sees the code on screen.]**
>
> *"Now, let us examine the critical code logics that give our platform its mathematical rigor. I will highlight five specific sections in the code:
>
> **Logic 1: Zero Data Leakage Split-Before-Fit Pipeline (`src/data_loader.py` lines 175–195 & `src/pipeline.py` lines 58–62)**
> [Highlight lines 175–195 in `data_loader.py`]
> Notice this code block: we split raw data into 70% Train, 10% Validation, and 20% Test using `stratify=df['emi_eligibility']`. In `pipeline.py`, notice that we call `preprocessor.fit_transform(train_feat)` strictly on the training partition! For test evaluation and live inference, we ONLY invoke `preprocessor.transform()`. This guarantees that test distribution parameters—such as imputation medians and standard deviation scalers—never contaminate model training.
>
> **Logic 2: Domain-Informed Outlier Capper with Non-Negativity Floors (`src/preprocessing.py` lines 34–66)**
> [Open `preprocessing.py`, highlight lines 49–57]
> Standard IQR capping at 1.5× can calculate negative lower bounds for positive financial variables. Notice lines 55–57 in our custom `OutlierCapper`: `lower = max(0.0, lower)`. For financial features like `monthly_salary`, `requested_amount`, and `current_emi_amount`, we strictly enforce an absolute zero floor. This preserves physical financial reality while clipping extreme tail outliers using the 1st and 99th percentiles.
>
> **Logic 3: Mathematical Free Cash Flow & 4-Factor Financial Health Scoring (`src/feature_engineering.py` lines 40–77)**
> [Open `feature_engineering.py`, highlight lines 40–77]
> Look at line 46: we compute `disposable_income = salary - total_monthly_expenses`, where total expenses sums rent, utilities, school fees, and current EMIs. Then, at lines 66–77, we compute our composite 0–100 Financial Health Score: credit score contributes 40 points, Debt-to-Income ratio contributes 25 points, disposable income buffer contributes 20 points, and employment stability contributes 15 points. This single feature captures over 21% of XGBoost's decision weight.
>
> **Logic 4: XGBoost Objective & Regularized Tree Tuning (`src/classification_models.py` lines 40–49 & `src/regression_models.py` lines 37–46)**
> [Open `classification_models.py`, highlight lines 40–49]
> Notice our hyperparameters: `max_depth=6`, `learning_rate=0.08`, `subsample=0.85`, and `colsample_bytree=0.85` with `eval_metric='mlogloss'`. We utilize 2nd-order Taylor expansions of the loss function, and by subsampling 85% of features per tree split, we completely eliminate multicollinearity between salary and disposable income.
>
> **Logic 5: Real-Time Dual-Target Inference & Underwriter Risk Rules (`src/predict.py` lines 29–85)**
> [Open `predict.py`, highlight lines 29–85]
> In our `predict_single()` method, a single JSON applicant payload is transformed through the fitted pipeline, evaluated by the classifier for multi-class risk, and sized by the regressor for maximum safe EMI. Then, lines 59–85 apply automated underwriter decision logic—generating human-readable audit flags whenever credit score is under 650, DTI exceeds 40%, or living expenses exceed cash flow."*

---

### Part 4: Scientific Hypothesis Testing Battery (09:00 – 11:00 | 2.0 Min)
> **[SCREEN ACTION: In VS Code, open `docs/chi_square_tests.csv` or show the EDA page in the browser under Tab 3.]**
>
> *"Before training any machine learning models, we conducted a rigorous hypothesis testing battery using SciPy. In machine learning engineering, algorithms should never be trained on assumed feature relevance—every predictor must be statistically proven:
>
> 1. **Chi-Square ($\chi^2$) Test of Independence for Categorical Variables:**
>    We formulated the Null Hypothesis ($H_0$) that categorical predictors—such as company type, employment type, and lending scenario—are independent of loan eligibility. The Chi-Square test yielded $p$-values strictly less than **0.001** for all primary categories. For example, company type produced a $\chi^2$ statistic of over 1,200 with $p < 0.0001$. We rejected the Null Hypothesis, confirming that employment stability and company stature have a statistically significant dependence with repayment behavior.
>
> 2. **One-Way ANOVA ($F$-Tests) for Continuous Variables:**
>    For continuous features, our Null Hypothesis ($H_0$) stated that the mean values of variables like disposable income, credit score, and monthly salary are identical across the three eligibility classes (Eligible, High_Risk, and Not_Eligible). The ANOVA test yielded massive $F$-statistics exceeding **450.0** with $p$-values of **0.000**. This proved that our risk tiers represent genuinely distinct mathematical populations, establishing the foundation for high classification accuracy.
>
> 3. **Pearson & Spearman Rank Correlations:**
>    For continuous EMI regression, our correlation analysis confirmed that engineered **Disposable Free Cash Flow** holds the strongest positive correlation with safe monthly EMI capacity at **$r = +0.92$**, outperforming raw gross salary at $r = +0.88$. This empirically validates our core premise: free cash flow pays loans, not gross income."*

---

### Part 5: Model Tournament & Empirical Results Deep Dive (11:00 – 14:00 | 3.0 Min)
> **[SCREEN ACTION: Open `models/XGBoost_confusion_matrix.png` and `models/XGBoost_Regressor_actual_vs_predicted.png`, or show them in the Model Evaluation tab of Streamlit.]**
>
> *"Now let us examine the core machine learning results and our model comparison tournament. To satisfy both tasks, we trained and compared three candidate models per problem type, tracking all runs inside MLflow:
>
> **Multi-Class Eligibility Classification Tournament:**
> We began with our linear baseline: **Logistic Regression** with $L_2$ regularization, achieving 85.97% accuracy and 0.8674 weighted F1. It struggled because credit default involves non-linear debt thresholds. Next, **Random Forest Classifier** achieved 97.32% accuracy through parallel bagged decision trees. Finally, our champion, **XGBoost Classifier**, achieved an outstanding **98.95% Accuracy, 0.9895 F1-Score, and 0.9998 ROC-AUC**, crushing the capstone requirement of 90%.
>
> **Confusion Matrix Analysis:** [Point to Confusion Matrix heatmap]
> Most importantly from a banking solvency standpoint, look at the confusion matrix diagonal: the misclassification between **Eligible and Not_Eligible is exactly 0.00%**. Our model never approves an uncreditworthy borrower, completely protecting the bank from catastrophic capital loss. The minor 1% errors occur strictly on the boundary between High_Risk and neighboring classes, which in banking practice are routed to human senior underwriters for secondary review.
>
> **Maximum Safe EMI Continuous Regression Tournament:**
> For continuous EMI prediction, **Linear Ridge Regression** served as our baseline with an RMSE of ₹2,641.46 and an R² of 0.9475—failing our ₹2,000 threshold because cash flow capacity drops abruptly when DTI crosses 40%. **Random Forest Regressor** reached an RMSE of ₹294.60. Decisively winning the tournament, **XGBoost Regressor achieved an RMSE of just ₹273.92, an MAE of ₹182.45, and an R² of 0.9994**.
>
> **Parity Plot & Homoscedasticity:** [Point to Parity scatter plot]
> On our actual vs predicted parity plot, the test predictions form a knife-edge ribbon adhering directly to the 45-degree line across the entire ₹500 to ₹50,000 range. Notice that the error variance does not fan out at higher loan amounts, confirming homoscedastic error distribution across all income brackets.
>
> **Feature Importance Synergy:**
> Our Gini gain feature importances reveal the dual logic of lending: **Classification** is dominated by risk metrics—Debt-to-Income ratio (28%) and Financial Health Score (21%). **Regression** is dominated by capacity metrics—Disposable Cash Flow (35%) and Monthly Salary (27%). In short: risk metrics decide *whether* we lend; cash flow decides *how much* we can safely lend."*

---

### Part 6: Live Streamlit Application Demo & Relative Graph Walkthrough (14:00 – 16:45 | 2.75 Min)
> **[SCREEN ACTION: Switch browser window to the live running Streamlit app at `http://localhost:8501`. Open Page 1: `Predict EMI`.]**
>
> *"Now, let us experience the live production platform deployed on Streamlit. I navigate to Page 1: the **Real-Time Underwriting Studio**.
>
> [Click '🌟 Persona 1: Reeva Shah' and click 'Run Underwriting Assessment']
> To demonstrate real-time dual inference, I select our pre-configured prime profile: **Reeva Shah**, a 34-year-old Senior MNC Lead earning ₹95,000 per month with an existing ₹8,000 EMI, an excellent 780 CIBIL score, requesting a ₹5,00,000 Vehicle Loan over 48 months. In under 200 milliseconds, both production models execute:
>
> Look at the results: the classifier awards an immediate **ELIGIBLE** status with 98.4% model confidence. Simultaneously, the regressor determines her **Maximum Safe EMI capacity at ₹15,996 per month**, which easily accommodates her requested installment of ₹10,416 per month.
>
> [Scroll down to the Plotly Visual Gauges]
> Notice our dynamic Plotly visual gauges: her **Debt-to-Income (DTI) ratio is just 8.42%**, sitting comfortably inside the safe green zone well below the 40% threshold. Her **Financial Health Score is 84 out of 100**, reflecting strong liquidity and career longevity.
>
> [Scroll to the Waterfall Cash Flow Diagram]
> Here, the waterfall diagram visualizes her monthly cash journey: starting from ₹95,000 gross salary, subtracting her ₹8,000 current EMI and ₹26,000 living expenses, leaving an unencumbered free cash buffer of **₹61,000**. This proves why she is a prime credit risk.
>
> [Briefly demonstrate Persona 2: Priya Patel and Persona 3: Vikram Singh]
> If we select **Priya Patel** (₹42K salary, 620 CIBIL), the model triggers a **High-Risk conditional approval** with tenure restructuring. When we select **Vikram Singh** (₹30K salary, 510 CIBIL, ₹9K debt), the model automatically declines the application due to a monthly cash flow deficit of -₹14,500. This proves the system is dynamic, sensitive, and mathematically calibrated."*

---

### Part 7: Future Expansion Roadmap & Concluding Defense (16:45 – 18:30 | 1.75 Min)
> **[SCREEN ACTION: Switch back to camera or show the 4-Pillar Architecture Diagram in the browser.]**
>
> *"**How would I expand EMIPredict AI in a production enterprise environment?** While our current dual-model architecture achieves state-of-the-art accuracy, I have designed a **Four-Pillar Engineering Roadmap** to scale this platform for tier-1 commercial banking:
>
> 1. **Account Aggregator (AA) Integration:** Connecting with RBI-regulated Account Aggregator NBFCs to ingest verified GST invoices and bank statements directly via Open Banking APIs, eliminating self-reported data bias.
> 2. **Explainable AI & Adverse Action Generation:** Implementing TreeSHAP to generate instant local feature waterfall charts for every declined applicant, auto-generating compliant FCRA/RBI Adverse Action Notices.
> 3. **Event-Driven Streaming via Apache Kafka:** Containerizing inference into FastAPI microservices subscribed to Kafka topics, capable of handling 10,000+ loan evaluations per second during peak festive e-commerce sales.
> 4. **Continuous MLOps Drift Monitoring via Evidently AI:** Tracking Kolmogorov-Smirnov data drift on macroeconomic inflation and interest rates, automatically triggering Airflow retraining pipelines when drift exceeds 0.15.
>
> In conclusion, EMIPredict AI replaces outdated, manual loan underwriting with mathematical cash flow intelligence—achieving **98.95% classification accuracy**, an **RMSE of ₹273.92**, and complete experiment lineage via MLflow.
> Thank you, respected evaluators and mentor, for your time and guidance. I am now ready for your questions!"*

---

## 📊 Part 4: Evaluator Visual Defense — Graph-by-Graph Walkthrough Script

When evaluators ask you to walk through the dashboard visuals or explain specific plots, use these exact, high-scoring verbal walkthroughs:

### A. Exploratory Data Analysis (EDA) Charts (`app/pages/2_EDA_Insights.py`)

#### Graph 1: Stacked Bar Chart — Eligibility Breakdown by Lending Scenario
* **Where to find it:** Streamlit Page 2 (`EDA Insights`) ➔ Tab 2: *Scenario & Demographic Patterns*.
* **What to say aloud:**
  > *"Respected evaluators, this stacked bar chart demonstrates that credit risk is fundamentally product-dependent. Notice that **E-commerce EMI** exhibits the highest approval rate at approximately 72%. This is because ticket sizes are modest (₹10,000 to ₹50,000), which easily fit within everyday disposable income. Conversely, **Vehicle and Personal Loans** show a substantial 45% to 50% concentration in High-Risk and Not-Eligible categories because they demand large principal outlays (up to ₹15 Lakhs), pushing borrowers' Debt-to-Income (DTI) ratio above the statutory 40% safety threshold."*
* **Examiner Trap Question:** *"Why not train separate models for each scenario?"*
* **Winning Answer:** *"Training a unified model with scenario one-hot features allowed the gradient booster to learn cross-product credit behavior while preserving the full statistical power of all 400,000 samples."*

#### Graph 2: Box Plot — Requested Loan Amount Distribution (Log-Scale INR)
* **Where to find it:** Streamlit Page 2 (`EDA Insights`) ➔ Tab 2: *Scenario & Demographic Patterns*.
* **What to say aloud:**
  > *"Here, we plotted requested loan amounts on a **logarithmic scale** because ticket sizes span three orders of magnitude—from ₹10,000 consumer electronics up to ₹15 Lakh automobiles. The horizontal line inside each box marks the median ticket size. Notice how **Vehicle EMI** displays both the highest median and the widest interquartile range (IQR), reflecting high product heterogeneity (ranging from entry two-wheelers to premium passenger vehicles). The outlier dots at the upper tail justify why we implemented **IQR Winsorization capping at the 99th percentile** to prevent extreme loans from skewing gradient calculations."*
* **Examiner Trap Question:** *"Why was a logarithmic scale strictly necessary here?"*
* **Winning Answer:** *"A linear scale would compress the ₹10K-₹50K e-commerce boxes into an invisible 1-millimeter sliver at the bottom, hiding the internal variance of lower-ticket loans."*

#### Graph 3: KDE Density Plot — Credit Score Distribution Across Eligibility Status
* **Where to find it:** Streamlit Page 2 (`EDA Insights`) ➔ Tab 3: *Risk & Credit Correlations*.
* **What to say aloud:**
  > *"This Kernel Density Estimation (KDE) plot proves the discriminative power of bureau credit scores. Notice the distinct **trimodal distribution**: prime eligible applicants (the green curve) cluster tightly above 750 with minimal density below 650. Conversely, ineligible applicants (the red curve) peak sharply below 580, indicating past delinquencies. Crucially, the amber curve represents our **High-Risk transition zone** between 620 and 710. This is exactly where static threshold underwriting fails and our ML model excels by evaluating debt cushion and cash flow to grant conditional approvals."*
* **Examiner Trap Question:** *"If credit scores separate classes so cleanly, why not just use an 'IF score > 700' rule?"*
* **Winning Answer:** *"Because credit score reflects past repayment, not current cash flow. A borrower with an 800 score who recently incurred ₹80,000 in fresh EMIs has great history but zero liquidity to service another loan."*

#### Graph 4: Bivariate Scatter Plot — Monthly Salary vs Safe Max EMI (Colored by Credit Score)
* **Where to find it:** Streamlit Page 2 (`EDA Insights`) ➔ Tab 3: *Risk & Credit Correlations*.
* **What to say aloud:**
  > *"This scatter plot illustrates how safe repayment capacity scales with income and credit quality. As monthly salary increases along the horizontal axis, the maximum safe EMI scales upward along a clean linear trajectory. Notice the vertical color stratification: at any identical salary level—say, ₹80,000 per month—applicants with yellow/green dots (credit scores > 750) are granted higher safe EMIs (₹25,000 to ₹30,000) than applicants with dark purple dots (credit scores < 600), whose capacity is capped at ₹10,000 to ₹15,000 due to risk premiums."*
* **Examiner Trap Question:** *"Why does the scatter width fan out (widen) at salaries above ₹1,00,000?"*
* **Winning Answer:** *"High earners exhibit greater lifestyle divergence: an executive with multiple prior mortgages has far less free cash flow than an executive with identical pay and zero debt. The model captures this variance."*

#### Graph 5: Pearson Correlation Heatmap — Key Financial Predictors vs Safe EMI
* **Where to find it:** Streamlit Page 2 (`EDA Insights`) ➔ Tab 3: *Risk & Credit Correlations*.
* **What to say aloud:**
  > *"This correlation heatmap provides an empirical audit of our feature space. Notice that our engineered feature, **Disposable Free Cash Flow**, has the highest positive correlation with safe EMI at **r = +0.92**, surpassing raw monthly salary at **r = +0.88**. This statistically validates our hypothesis that unencumbered cash surplus predicts repayment capacity better than gross income. Furthermore, `current_emi_amount` displays a clear negative correlation (-0.38), confirming that existing debt burdens actively compress repayment headroom."*
* **Examiner Trap Question:** *"Salary and disposable income have high collinearity (r > 0.85). Did this degrade your models?"*
* **Winning Answer:** *"Collinearity hurts ordinary linear regression, which is why Linear Ridge suffered (RMSE ₹2,641). However, tree-based models like Random Forest and XGBoost evaluate splits orthogonally and are completely immune to collinearity."*

---

### B. Model Performance & Diagnostic Visuals (`app/pages/3_Model_Evaluation.py` & `models/`)

#### Graph 6: Confusion Matrix Heatmap — XGBoost Multi-Class Classifier
* **Where to find it:** Streamlit Page 3 (`Model Evaluation`) ➔ Tab 3: *Artifacts & Visualizations*.
* **What to say aloud:**
  > *"Respected evaluators, this confusion matrix confirms the exceptional **98.95% accuracy** of our champion XGBoost Classifier. The deep blue diagonal represents true positives across all three tiers. Most crucially from a credit risk standpoint, look at the corner cells: the confusion between **Eligible and Not_Eligible is exactly 0.00%**. The model never misclassifies an uncreditworthy borrower as prime eligible, eliminating catastrophic loan defaults. The only minor classification boundary errors—less than 1%—occur between High_Risk and its neighbors, which in banking are routed for manual underwriter review anyway."*
* **Examiner Trap Question:** *"Why do you prioritize Precision over Recall on the 'Eligible' class?"*
* **Winning Answer:** *"Because in credit underwriting, a False Positive (lending to a defaulter) destroys 100% of the principal loan balance. A False Negative only loses 5-10% in prospective interest. High precision protects the bank's capital solvency."*

#### Graph 7: Feature Importance Bar Chart — XGBoost Classifier (Gini Gain)
* **Where to find it:** Streamlit Page 3 (`Model Evaluation`) ➔ Tab 3: *Artifacts & Visualizations*.
* **What to say aloud:**
  > *"This feature importance chart illustrates what actually drives loan eligibility. Notice that our engineered feature, **Debt-to-Income (DTI) ratio**, ranks as the #1 most decisive predictor, accounting for over 28% of total split gain, followed by our **Financial Health Score** at 21% and **Credit Score** at 16%. This proves a core credit risk reality: loan default is governed by relative leverage (how much debt an applicant carries relative to their earnings) rather than absolute salary. Gross income ranks lower because high earners with heavy debt default just as frequently as low earners."*
* **Examiner Trap Question:** *"Why did you use Gain instead of Split Count or Permutation Importance?"*
* **Winning Answer:** *"Split count is biased toward continuous high-cardinality features that split often with tiny improvements. Gain measures the actual reduction in cross-entropy loss, giving a pure, uninflated measure of predictive power."*

#### Graph 8: Parity Scatter Plot — XGBoost Regressor Actual vs Predicted Safe EMI (45° Line)
* **Where to find it:** Streamlit Page 3 (`Model Evaluation`) ➔ Tab 3: *Artifacts & Visualizations*.
* **What to say aloud:**
  > *"This parity plot visualizes the precision of our continuous regression engine. The dashed diagonal marks the line of perfect agreement ($y = x$). Notice how our test set predictions form an extremely tight, knife-edge ribbon adhering directly to the 45-degree line across the entire ₹500 to ₹50,000 range. This tight alignment explains our **$R^2$ score of 0.9994** and an **RMSE of just ₹273.92**—which crushes the capstone requirement of ₹2,000. Furthermore, the residual variance does not expand at higher ticket sizes, demonstrating homoscedastic errors across all income brackets."*
* **Examiner Trap Question:** *"An R² of 0.9994 seems abnormally high. Did the model memorize the data?"*
* **Winning Answer:** *"No, because this $R^2$ was computed strictly on the unseen 20% holdout test set with zero data leakage. Safe monthly debt capacity is bounded by deterministic financial physics (Income minus Debt minus Living Costs), which tree ensembles learn almost perfectly once non-linearities are captured."*

#### Graph 9: Feature Importance Bar Chart — XGBoost Regressor (Gain-Based)
* **Where to find it:** Streamlit Page 3 (`Model Evaluation`) ➔ Tab 3: *Artifacts & Visualizations*.
* **What to say aloud:**
  > *"Comparing this chart to our classification feature importance reveals the core financial synergy of our platform. While classification was driven by credit risk indicators (DTI and Credit Score), regression is dominated by **Net Disposable Free Cash Flow (35%)** and **Monthly Salary (27%)**. This aligns with fundamental banking mechanics: creditworthiness determines *whether* we lend, but unencumbered monthly cash surplus determines *how much* we can safely lend without triggering default."*
* **Examiner Trap Question:** *"Why does Credit Score rank lower in the regression model?"*
* **Winning Answer:** *"Credit score acts as a qualitative trust gatekeeper, but does not pay bills. The rupee installment must be serviced from physical liquid cash flow every month, which is why cash flow features dominate sizing."*

#### Graph 10: Model Comparison Tournament Leaderboard
* **Where to find it:** Streamlit Page 3 (`Model Evaluation`) ➔ Tabs 1 & 2: *Classification Models* & *Regression Models*.
* **What to say aloud:**
  > *"This comparison leaderboard documents our systematic experimentation logged in MLflow. We began with baseline linear models: Logistic Regression achieved 85.97% accuracy, while Linear Ridge had an RMSE of ₹2,641.46. Both failed to model non-linear debt cliffs. Random Forest produced a dramatic leap forward by capturing orthogonal splits, achieving 97.32% accuracy and ₹294.60 RMSE. Finally, **XGBoost decisively won both tasks** (98.95% accuracy; ₹273.92 RMSE) because its 2nd-order Taylor series gradient optimization and column subsampling allow it to fit complex interactions while strictly penalizing tree complexity."*
* **Examiner Trap Question:** *"Why did you use MLflow to track these instead of just writing metrics to a CSV?"*
* **Winning Answer:** *"MLflow provides immutable, auditable experiment lineage—logging git commit hashes, hyperparameters, dataset hashes, and serialized model binaries. In banking, regulatory compliance requires full auditability for every deployed credit model."*
