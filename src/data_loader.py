"""
EMIPredict AI - Data Loader Module
Handles data ingestion, validation, splitting, and scenario partitioning.
Includes an auto-detection utility for downloaded files.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, RANDOM_STATE,
    TEST_SIZE, VALIDATION_SIZE, ALL_INPUT_FEATURES,
    CLASSIFICATION_TARGET, REGRESSION_TARGET, EMI_SCENARIOS
)


def find_and_stage_raw_dataset() -> Path:
    """
    Checks if the dataset is in data/raw or in the user's Downloads folder.
    Copies to data/raw/emi_prediction_dataset.csv if found in Downloads.
    """
    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 100_000:
        return RAW_DATA_PATH
        
    # Search common download locations
    downloads_dir = Path.home() / "Downloads"
    candidates = list(downloads_dir.glob("*emi*prediction*.csv")) + list(downloads_dir.glob("*emi*.csv"))
    
    for candidate in candidates:
        if candidate.stat().st_size > 10_000_000:  # > 10MB
            print(f"Found dataset at {candidate}, staging to {RAW_DATA_PATH}...")
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate, RAW_DATA_PATH)
            return RAW_DATA_PATH

    return RAW_DATA_PATH


def generate_benchmark_synthetic_data(num_samples: int = 10000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a highly realistic synthetic dataset conforming precisely to the 
    project document schema and distributions across the 5 EMI scenarios.
    Useful for pipeline verification, testing, and continuous integration.
    """
    np.random.seed(random_state)
    records_per_scenario = num_samples // len(EMI_SCENARIOS)
    
    dfs = []
    scenario_params = {
        "E-commerce Shopping EMI": {"amount": (10000, 200000), "tenure": (3, 24)},
        "Home Appliances EMI": {"amount": (20000, 300000), "tenure": (6, 36)},
        "Vehicle EMI": {"amount": (80000, 1500000), "tenure": (12, 84)},
        "Personal Loan EMI": {"amount": (50000, 1000000), "tenure": (12, 60)},
        "Education EMI": {"amount": (50000, 500000), "tenure": (6, 48)},
    }
    
    for scenario, params in scenario_params.items():
        n = records_per_scenario
        age = np.random.randint(25, 61, size=n)
        gender = np.random.choice(["Male", "Female"], size=n, p=[0.65, 0.35])
        marital_status = np.random.choice(["Single", "Married"], size=n, p=[0.4, 0.6])
        education = np.random.choice(["High School", "Graduate", "Post Graduate", "Professional"], 
                                     size=n, p=[0.15, 0.50, 0.25, 0.10])
        monthly_salary = np.random.randint(15000, 200001, size=n)
        employment_type = np.random.choice(["Private", "Government", "Self-employed"], 
                                           size=n, p=[0.65, 0.20, 0.15])
        years_of_employment = np.clip(np.random.exponential(scale=5.0, size=n), 0, age - 20)
        company_type = np.random.choice(["Startup", "Mid-sized", "Large Enterprise", "MNC", "Government"], 
                                        size=n, p=[0.25, 0.30, 0.20, 0.15, 0.10])
        house_type = np.random.choice(["Rented", "Own", "Family"], size=n, p=[0.45, 0.35, 0.20])
        
        # Expenses based on salary and house type
        monthly_rent = np.where(house_type == "Rented", np.clip(monthly_salary * np.random.uniform(0.15, 0.35, n), 3000, 45000), 0.0)
        family_size = np.random.randint(1, 7, size=n)
        dependents = np.clip(family_size - np.random.randint(1, 3, size=n), 0, 4)
        
        school_fees = np.where(dependents > 0, np.clip(monthly_salary * np.random.uniform(0.05, 0.15, n), 0, 25000), 0.0)
        college_fees = np.where(dependents > 1, np.clip(monthly_salary * np.random.uniform(0.05, 0.15, n), 0, 30000), 0.0)
        travel_expenses = np.clip(monthly_salary * np.random.uniform(0.03, 0.08, n), 1000, 15000)
        groceries_utilities = np.clip(monthly_salary * np.random.uniform(0.10, 0.25, n), 4000, 35000)
        other_monthly_expenses = np.clip(monthly_salary * np.random.uniform(0.02, 0.10, n), 1000, 20000)
        
        existing_loans = np.random.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
        current_emi_amount = np.where(existing_loans == "Yes", 
                                      np.clip(monthly_salary * np.random.uniform(0.10, 0.30, n), 2000, 40000), 0.0)
        
        credit_score = np.random.randint(300, 851, size=n)
        bank_balance = np.clip(monthly_salary * np.random.uniform(0.5, 4.0, n), 2000, 800000)
        emergency_fund = np.clip(monthly_salary * np.random.uniform(0.2, 3.0, n), 0, 600000)
        
        # Loan details
        req_amount = np.random.randint(params["amount"][0], params["amount"][1] + 1, size=n)
        req_tenure = np.random.randint(params["tenure"][0], params["tenure"][1] + 1, size=n)
        
        # Target Generation
        total_exp = (monthly_rent + school_fees + college_fees + travel_expenses + 
                     groceries_utilities + other_monthly_expenses + current_emi_amount)
        free_cash = monthly_salary - total_exp
        dti = current_emi_amount / monthly_salary
        req_emi = req_amount / req_tenure
        
        # Safe Max EMI: ~ 40-50% of free cash, capped at 50% salary - existing debt, bounded [500, 50000]
        max_monthly_emi = np.clip(
            np.maximum(500.0, free_cash * 0.45 * (credit_score / 750.0)),
            500.0, 50000.0
        ).round(2)
        
        # Eligibility Classification
        eligibility = []
        for i in range(n):
            if credit_score[i] >= 680 and free_cash[i] >= req_emi[i] * 1.2 and dti[i] < 0.40:
                eligibility.append("Eligible")
            elif credit_score[i] >= 580 and free_cash[i] >= req_emi[i] * 0.9 and dti[i] < 0.55:
                eligibility.append("High_Risk")
            else:
                eligibility.append("Not_Eligible")
                
        df_sub = pd.DataFrame({
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment.round(1),
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent.round(2),
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees.round(2),
            "college_fees": college_fees.round(2),
            "travel_expenses": travel_expenses.round(2),
            "groceries_utilities": groceries_utilities.round(2),
            "other_monthly_expenses": other_monthly_expenses.round(2),
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount.round(2),
            "credit_score": credit_score,
            "bank_balance": bank_balance.round(2),
            "emergency_fund": emergency_fund.round(2),
            "emi_scenario": scenario,
            "requested_amount": req_amount,
            "requested_tenure": req_tenure,
            "emi_eligibility": eligibility,
            "max_monthly_emi": max_monthly_emi
        })
        dfs.append(df_sub)
        
    return pd.concat(dfs, ignore_index=True)


def load_dataset(use_synthetic_if_missing: bool = True) -> pd.DataFrame:
    """
    Loads dataset from raw path if available, otherwise generates benchmark data.
    """
    raw_path = find_and_stage_raw_dataset()
    if raw_path.exists() and raw_path.stat().st_size > 100_000:
        print(f"Loading actual raw dataset from {raw_path}...")
        df = pd.read_csv(raw_path)
    elif use_synthetic_if_missing:
        print(f"Raw dataset not detected at {raw_path}. Generating benchmark 20,000 record dataset for immediate execution...")
        df = generate_benchmark_synthetic_data(num_samples=20000)
    else:
        raise FileNotFoundError(f"Dataset not found at {raw_path}")
        
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, val_size: float = VALIDATION_SIZE) -> Dict[str, pd.DataFrame]:
    """
    Creates stratified Train / Validation / Test splits preserving class balance.
    """
    # First split off test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[CLASSIFICATION_TARGET],
        random_state=RANDOM_STATE
    )
    
    # Next split train into train & validation
    val_rel_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_rel_size,
        stratify=train_val[CLASSIFICATION_TARGET],
        random_state=RANDOM_STATE
    )
    
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
