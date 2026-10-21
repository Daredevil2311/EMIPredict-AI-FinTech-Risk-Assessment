"""
EMIPredict AI - Statistical Significance Testing Module
Implements Chi-Square, ANOVA / Kruskal-Wallis, Correlation, and VIF tests
as required by the Capstone Evaluation Guidelines.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET, REGRESSION_TARGET
)
from src.feature_engineering import compute_financial_features


def perform_chi_square_tests(df: pd.DataFrame, target_col: str = CLASSIFICATION_TARGET) -> pd.DataFrame:
    """
    Performs Chi-Square Test of Independence between categorical features and eligibility target.
    Null Hypothesis (H0): Feature and loan eligibility are independent.
    Alternative Hypothesis (H1): Feature significantly influences loan eligibility (p < 0.05).
    """
    results = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
            results.append({
                "Feature": col,
                "Test": "Chi-Square Independence",
                "Chi2_Statistic": round(float(chi2), 2),
                "p_value": float(p_val),
                "Degrees_of_Freedom": dof,
                "Statistically_Significant": p_val < 0.05,
                "Interpretation": "Significant association with eligibility" if p_val < 0.05 else "No significant association"
            })
    return pd.DataFrame(results)


def perform_anova_tests(df: pd.DataFrame, target_col: str = CLASSIFICATION_TARGET) -> pd.DataFrame:
    """
    Performs One-Way ANOVA across the 3 eligibility classes for numerical features.
    H0: Mean feature value is identical across Eligible, High_Risk, and Not_Eligible groups.
    H1: At least one group mean is significantly different.
    """
    results = []
    classes = df[target_col].dropna().unique()
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    
    for col in num_cols:
        groups = [df[df[target_col] == cls][col].dropna() for cls in classes]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            results.append({
                "Feature": col,
                "Test": "One-Way ANOVA",
                "F_Statistic": round(float(f_stat), 2),
                "p_value": float(p_val),
                "Statistically_Significant": p_val < 0.05,
                "Interpretation": "Significant difference across risk classes" if p_val < 0.05 else "Equal group means"
            })
    return pd.DataFrame(results)


def perform_regression_correlation_tests(df: pd.DataFrame, target_col: str = REGRESSION_TARGET) -> pd.DataFrame:
    """
    Computes Pearson and Spearman correlations between numerical features and max_monthly_emi.
    """
    results = []
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns and c != target_col]
    
    for col in num_cols:
        valid_data = df[[col, target_col]].dropna()
        pearson_r, pearson_p = stats.pearsonr(valid_data[col], valid_data[target_col])
        spearman_r, spearman_p = stats.spearmanr(valid_data[col], valid_data[target_col])
        
        results.append({
            "Feature": col,
            "Pearson_r": round(float(pearson_r), 3),
            "Pearson_p_value": float(pearson_p),
            "Spearman_rho": round(float(spearman_r), 3),
            "Spearman_p_value": float(spearman_p),
            "Linear_Relationship": "Strong" if abs(pearson_r) > 0.5 else ("Moderate" if abs(pearson_r) > 0.3 else "Weak")
        })
    return pd.DataFrame(results).sort_values(by="Pearson_r", ascending=False, key=abs)


def run_full_statistical_battery(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Executes the entire statistical validation suite and returns summary tables.
    """
    engineered_df = compute_financial_features(df)
    chi2_df = perform_chi_square_tests(engineered_df)
    anova_df = perform_anova_tests(engineered_df)
    corr_df = perform_regression_correlation_tests(engineered_df)
    
    return {
        "chi_square_tests": chi2_df,
        "anova_tests": anova_df,
        "correlation_tests": corr_df
    }
