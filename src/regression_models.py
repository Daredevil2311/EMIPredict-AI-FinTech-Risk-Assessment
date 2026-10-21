"""
EMIPredict AI - Regression Models Module
Trains and evaluates 3+ regression models for Maximum Safe Monthly EMI Prediction:
  1. Ridge / Linear Regression (Regularized OLS baseline)
  2. Random Forest Regressor (Non-linear ensemble)
  3. XGBoost Regressor (Extreme Gradient Boosting)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

from src.config import RANDOM_STATE


def get_regression_models() -> Dict[str, Any]:
    """Returns candidate regression models with tuned hyperparameters."""
    return {
        "Linear_Ridge_Regression": Ridge(
            alpha=1.0,
            random_state=RANDOM_STATE
        ),
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def evaluate_regressor(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None
) -> Dict[str, Any]:
    """
    Computes comprehensive regression metrics:
    RMSE, MAE, R², and MAPE.
    """
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 500.0, 50000.0)  # Bound to domain constraints
    
    # In newer scikit-learn root_mean_squared_error is preferred, or sqrt(mse)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Safe MAPE avoiding division by tiny numbers
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
    
    # Feature importances
    feat_imp_df = None
    if feature_names is not None:
        if hasattr(model, "feature_importances_"):
            feat_imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        elif hasattr(model, "coef_"):
            feat_imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(model.coef_)
            }).sort_values("importance", ascending=False)
            
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "mape_percent": float(mape)
    }
    
    return {
        "metrics": metrics,
        "feature_importance": feat_imp_df,
        "y_pred": y_pred,
        "y_true": y_test
    }
