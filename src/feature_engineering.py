"""
EMIPredict AI - Feature Engineering Module
Computes domain-specific FinTech ratios, affordability metrics, and risk scores.
"""

import numpy as np
import pandas as pd


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived financial ratios and capacity indicators from raw financial variables.
    
    Derived Features:
      - total_monthly_expenses: Aggregate monthly obligations.
      - disposable_income: Net monthly cash flow after all obligations.
      - debt_to_income_ratio (DTI): Ratio of current debt EMI to gross monthly salary.
      - expense_to_income_ratio (ETI): Living obligations burden relative to salary.
      - requested_monthly_burden: Principal payment per tenure month.
      - requested_burden_ratio: Requested installment burden relative to salary.
      - savings_to_income_ratio: Total liquid cushion (bank + emergency) relative to salary.
      - financial_health_score: Multi-factor score (0-100) combining credit, liquidity, and leverage.
    """
    data = df.copy()
    
    # Fill NAs in financial amounts with 0 for safety
    expense_cols = [
        "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses"
    ]
    for c in expense_cols:
        if c not in data.columns:
            data[c] = 0.0
        else:
            data[c] = data[c].fillna(0.0)
            
    current_emi = data["current_emi_amount"].fillna(0.0) if "current_emi_amount" in data.columns else 0.0
    salary = data["monthly_salary"].replace(0, np.nan).fillna(1.0)
    
    # 1. Total Obligations
    data["total_monthly_expenses"] = (
        data[expense_cols].sum(axis=1) + current_emi
    )
    
    # 2. Disposable / Free Cash Flow
    data["disposable_income"] = data["monthly_salary"] - data["total_monthly_expenses"]
    
    # 3. Debt to Income Ratio (DTI)
    data["debt_to_income_ratio"] = (current_emi / salary).clip(0, 5)
    
    # 4. Expense to Income Ratio (ETI)
    data["expense_to_income_ratio"] = (data["total_monthly_expenses"] / salary).clip(0, 5)
    
    # 5. Requested Monthly Burden & Ratio
    req_amount = data["requested_amount"].fillna(0.0)
    req_tenure = data["requested_tenure"].replace(0, np.nan).fillna(12.0)
    data["requested_monthly_burden"] = req_amount / req_tenure
    data["requested_burden_ratio"] = (data["requested_monthly_burden"] / salary).clip(0, 5)
    
    # 6. Savings to Income Cushion
    bank_bal = data["bank_balance"].fillna(0.0)
    emerg_fund = data["emergency_fund"].fillna(0.0)
    data["savings_to_income_ratio"] = ((bank_bal + emerg_fund) / salary).clip(0, 50)
    
    # 7. Composite Financial Health Score (0 - 100)
    # Credit score contribution (0 - 40 pts)
    credit_pts = ((data["credit_score"].clip(300, 850) - 300) / 550.0) * 40.0
    # DTI score contribution (0 - 25 pts, lower is better)
    dti_pts = (1.0 - data["debt_to_income_ratio"].clip(0, 1)) * 25.0
    # Disposable Income cushion (0 - 20 pts)
    disp_ratio = (data["disposable_income"] / salary).clip(0, 1)
    disp_pts = disp_ratio * 20.0
    # Employment stability (0 - 15 pts)
    emp_years = data["years_of_employment"].clip(0, 15)
    emp_pts = (emp_years / 15.0) * 15.0
    
    data["financial_health_score"] = (credit_pts + dti_pts + disp_pts + emp_pts).clip(0, 100)
    
    return data
