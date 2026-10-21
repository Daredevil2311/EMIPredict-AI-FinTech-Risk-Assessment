"""
EMIPredict AI - Classification Models Module
Trains and evaluates 3+ classification models for EMI Eligibility Prediction:
  1. Logistic Regression (L2 regularized, class weighted)
  2. Random Forest Classifier (Ensemble bagged decision trees)
  3. XGBoost Classifier (Optimized extreme gradient boosting)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from src.config import RANDOM_STATE, CLASS_MAPPING, INVERSE_CLASS_MAPPING


def get_classification_models() -> Dict[str, Any]:
    """Returns the candidate classification models with tuned baseline hyperparameters."""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    try:
        if y_proba is not None and len(np.unique(y_test)) == y_proba.shape[1]:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            auc = 0.0
    except Exception:
        auc = 0.0
        
    cm = confusion_matrix(y_test, y_pred)
    
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
                "importance": np.mean(np.abs(model.coef_), axis=0)
            }).sort_values("importance", ascending=False)
            
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_ovr": float(auc)
    }
    
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, target_names=["Eligible", "High_Risk", "Not_Eligible"]),
        "feature_importance": feat_imp_df,
        "y_pred": y_pred,
        "y_proba": y_proba
    }
