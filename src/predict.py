"""
EMIPredict AI - Real-Time Inference Module
Loads serialized preprocessors and best models to serve predictions with automated risk explanations.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, INVERSE_CLASS_MAPPING
from src.feature_engineering import compute_financial_features


class EMIPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.preprocessor_path = models_dir / "preprocessor.joblib"
        self.classifier_path = models_dir / "best_classifier.joblib"
        self.regressor_path = models_dir / "best_regressor.joblib"
        
        if not (self.preprocessor_path.exists() and self.classifier_path.exists() and self.regressor_path.exists()):
            raise FileNotFoundError("Trained models not found. Run pipeline.py first.")
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.classifier = joblib.load(self.classifier_path)
        self.regressor = joblib.load(self.regressor_path)
        
    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs dual prediction (classification + regression) for an individual applicant."""
        df = pd.DataFrame([applicant_data])
        df_feat = compute_financial_features(df)
        
        X_trans = self.preprocessor.transform(df_feat)
        
        # Classification
        pred_class_idx = int(self.classifier.predict(X_trans)[0])
        pred_eligibility = INVERSE_CLASS_MAPPING.get(pred_class_idx, "Unknown")
        
        class_probas = {}
        if hasattr(self.classifier, "predict_proba"):
            probas = self.classifier.predict_proba(X_trans)[0]
            for idx, prob in enumerate(probas):
                label = INVERSE_CLASS_MAPPING.get(idx, f"Class_{idx}")
                class_probas[label] = round(float(prob) * 100, 2)
                
        # Regression
        pred_max_emi = float(self.regressor.predict(X_trans)[0])
        pred_max_emi = round(float(np.clip(pred_max_emi, 500.0, 50000.0)), 2)
        
        # Financial Ratios & AI Underwriter Recommendation
        dti = round(float(df_feat["debt_to_income_ratio"].iloc[0]) * 100, 2)
        eti = round(float(df_feat["expense_to_income_ratio"].iloc[0]) * 100, 2)
        disposable = round(float(df_feat["disposable_income"].iloc[0]), 2)
        req_burden = round(float(df_feat["requested_monthly_burden"].iloc[0]), 2)
        health_score = round(float(df_feat["financial_health_score"].iloc[0]), 1)
        
        # Underwriter Decision Logic
        risk_flags = []
        if applicant_data.get("credit_score", 700) < 650:
            risk_flags.append("Credit score below standard benchmark (650)")
        if dti > 40.0:
            risk_flags.append(f"Existing Debt-to-Income is elevated at {dti}%")
        if req_burden > pred_max_emi:
            risk_flags.append(f"Requested installment (INR {req_burden:,.2f}) exceeds max safe EMI (INR {pred_max_emi:,.2f})")
        if disposable < req_burden:
            risk_flags.append(f"Disposable cash flow (INR {disposable:,.2f}) is insufficient for requested EMI")
            
        if pred_eligibility == "Eligible":
            recommendation = (
                f"APPROVED: Strong financial profile. Applicant can comfortably service up to "
                f"INR {pred_max_emi:,.2f}/month. Proceed with standard prime interest rate."
            )
        elif pred_eligibility == "High_Risk":
            recommendation = (
                f"CONDITIONAL APPROVAL: Elevated risk profile. Recommend capping monthly EMI to "
                f"INR {pred_max_emi:,.2f}/month, requiring a co-applicant, or applying risk-adjusted interest rates."
            )
        else:
            recommendation = (
                f"REJECTED: Substantial default risk. Current financial obligations and debt ratios "
                f"impair repayment capacity. Recommend debt consolidation or increasing tenure."
            )
            
        return {
            "eligibility": pred_eligibility,
            "probabilities": class_probas,
            "max_safe_monthly_emi": pred_max_emi,
            "financial_health_score": health_score,
            "disposable_income": disposable,
            "debt_to_income_pct": dti,
            "expense_to_income_pct": eti,
            "requested_monthly_burden": req_burden,
            "risk_flags": risk_flags,
            "recommendation": recommendation
        }
