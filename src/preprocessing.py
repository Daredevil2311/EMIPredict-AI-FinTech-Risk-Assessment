"""
EMIPredict AI - Preprocessing Pipeline Module
Builds production-ready scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DERIVED_FEATURES
from src.feature_engineering import compute_financial_features


class FinancialFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to integrate feature engineering into a Scikit-Learn Pipeline."""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return compute_financial_features(X)
        elif isinstance(X, dict):
            return compute_financial_features(pd.DataFrame([X]))
        else:
            raise TypeError("Input must be a pandas DataFrame or dict")


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numerical outliers using domain thresholds and IQR bounds."""
    
    def __init__(self, factor: float = 3.0):
        self.factor = factor
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        for col in X_df.columns:
            if np.issubdtype(X_df[col].dtype, np.number):
                q25 = X_df[col].quantile(0.01)
                q75 = X_df[col].quantile(0.99)
                iqr = q75 - q25
                lower = q25 - (self.factor * iqr)
                upper = q75 + (self.factor * iqr)
                # Safeguards for strictly positive financial fields
                if col in ["monthly_salary", "requested_amount", "current_emi_amount"]:
                    lower = max(0.0, lower)
                self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df


def build_preprocessor(all_num_cols: Optional[List[str]] = None,
                       cat_cols: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer with imputation, scaling, and encoding.
    """
    if all_num_cols is None:
        all_num_cols = NUMERICAL_FEATURES + DERIVED_FEATURES
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES

    # Numerical pipeline: Median Impute -> StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: Most Frequent Impute -> One-Hot Encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor
