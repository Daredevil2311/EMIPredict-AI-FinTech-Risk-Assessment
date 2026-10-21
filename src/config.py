"""
EMIPredict AI - Configuration Module
Defines schema, hyperparameters, file paths, and target variables.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "emi_prediction_dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "docs"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Ensure directories exist
for directory in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, MLFLOW_ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Column Definitions
NUMERICAL_FEATURES = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
]

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]

DERIVED_FEATURES = [
    "total_monthly_expenses",
    "disposable_income",
    "debt_to_income_ratio",       # DTI = current_emi / monthly_salary
    "expense_to_income_ratio",    # ETI = total_expenses / monthly_salary
    "requested_monthly_burden",   # requested_amount / requested_tenure
    "requested_burden_ratio",     # requested_burden / monthly_salary
    "savings_to_income_ratio",    # (bank_balance + emergency_fund) / monthly_salary
    "financial_health_score",     # Composite risk score
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Target Variables
CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

# Class Mapping for Eligibility
CLASS_MAPPING = {
    "Eligible": 0,
    "High_Risk": 1,
    "Not_Eligible": 2
}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Lending Scenarios
EMI_SCENARIOS = [
    "E-commerce Shopping EMI",
    "Home Appliances EMI",
    "Vehicle EMI",
    "Personal Loan EMI",
    "Education EMI"
]

# Random State & Splitting
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Evaluation Targets
BENCHMARK_CLASSIFICATION_ACCURACY = 0.90
BENCHMARK_REGRESSION_RMSE = 2000.0
