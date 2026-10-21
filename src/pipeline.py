"""
EMIPredict AI - Master Training & Experimentation Pipeline
Executes end-to-end data processing, statistical validation, model training,
MLflow experiment tracking, and artifact persistence.
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR, REPORTS_DIR, CLASSIFICATION_TARGET,
    REGRESSION_TARGET, CLASS_MAPPING, INVERSE_CLASS_MAPPING,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DERIVED_FEATURES
)
from src.data_loader import load_dataset, split_data
from src.feature_engineering import compute_financial_features
from src.preprocessing import build_preprocessor
from src.statistical_tests import run_full_statistical_battery
from src.classification_models import get_classification_models, evaluate_classifier
from src.regression_models import get_regression_models, evaluate_regressor
from src.mlflow_utils import log_classification_run, log_regression_run, setup_mlflow_experiment


def run_pipeline():
    print("=" * 70)
    print("[*] STARTING EMIPredict AI END-TO-END PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    raw_df = load_dataset(use_synthetic_if_missing=True)
    
    # 2. Statistical Significance Testing
    print("\n[*] Running Statistical Significance Testing Battery...")
    stat_results = run_full_statistical_battery(raw_df)
    for test_name, res_df in stat_results.items():
        out_csv = REPORTS_DIR / f"{test_name}.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"  Saved {test_name} results to {out_csv}")
        
    # 3. Train-Val-Test Stratified Split
    print("\n[*] Creating Stratified Train / Validation / Test Splits...")
    splits = split_data(raw_df)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows")
    
    # 4. Feature Engineering on Splits
    print("\n[*] Computing Domain-Specific Financial Ratios...")
    train_feat = compute_financial_features(train_df)
    val_feat = compute_financial_features(val_df)
    test_feat = compute_financial_features(test_df)
    
    # 5. Fit Preprocessor (Imputation + OneHot + Scaling)
    print("\n[*] Fitting Preprocessing Pipeline...")
    all_num_cols = NUMERICAL_FEATURES + DERIVED_FEATURES
    preprocessor = build_preprocessor(all_num_cols, CATEGORICAL_FEATURES)
    
    X_train = preprocessor.fit_transform(train_feat)
    X_val = preprocessor.transform(val_feat)
    X_test = preprocessor.transform(test_feat)
    
    # Extract feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_feature_names = all_num_cols + encoded_cat_names
    
    # Targets
    y_train_clf = train_feat[CLASSIFICATION_TARGET].map(CLASS_MAPPING).values
    y_test_clf = test_feat[CLASSIFICATION_TARGET].map(CLASS_MAPPING).values
    
    y_train_reg = train_feat[REGRESSION_TARGET].values
    y_test_reg = test_feat[REGRESSION_TARGET].values
    
    # Setup MLflow
    setup_mlflow_experiment()
    
    # ----------------------------------------------------
    # 6. Classification Models
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("[*] TRAINING CLASSIFICATION MODELS (EMI Eligibility Prediction)")
    print("=" * 70)
    
    clf_models = get_classification_models()
    clf_results = {}
    best_clf_name = None
    best_clf_f1 = -1.0
    
    class_names = ["Eligible", "High_Risk", "Not_Eligible"]
    
    for name, model in clf_models.items():
        print(f"\nTraining Classification Model: {name}...")
        model.fit(X_train, y_train_clf)
        eval_res = evaluate_classifier(model, X_test, y_test_clf, all_transformed_feature_names)
        clf_results[name] = eval_res["metrics"]
        
        print(f"  Accuracy:  {eval_res['metrics']['accuracy'] * 100:.2f}%")
        print(f"  F1-Score:  {eval_res['metrics']['f1_weighted']:.4f}")
        print(f"  ROC-AUC:   {eval_res['metrics']['roc_auc_ovr']:.4f}")
        
        # Log to MLflow
        params = {k: str(v) for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))}
        log_classification_run(
            model_name=name,
            model=model,
            params=params,
            metrics=eval_res["metrics"],
            confusion_mat=eval_res["confusion_matrix"],
            class_names=class_names,
            feature_importance=eval_res["feature_importance"]
        )
        
        if eval_res["metrics"]["f1_weighted"] > best_clf_f1:
            best_clf_f1 = eval_res["metrics"]["f1_weighted"]
            best_clf_name = name
            best_clf_model = model
            best_clf_feat_imp = eval_res["feature_importance"]
            
    print(f"\n[WINNER] Best Classification Model Selected: {best_clf_name} (F1: {best_clf_f1:.4f})")
    
    # ----------------------------------------------------
    # 7. Regression Models
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("[*] TRAINING REGRESSION MODELS (Maximum Safe Monthly EMI)")
    print("=" * 70)
    
    reg_models = get_regression_models()
    reg_results = {}
    best_reg_name = None
    best_reg_rmse = float("inf")
    
    for name, model in reg_models.items():
        print(f"\nTraining Regression Model: {name}...")
        model.fit(X_train, y_train_reg)
        eval_res = evaluate_regressor(model, X_test, y_test_reg, all_transformed_feature_names)
        reg_results[name] = eval_res["metrics"]
        
        print(f"  RMSE: {eval_res['metrics']['rmse']:.2f} INR")
        print(f"  MAE:  {eval_res['metrics']['mae']:.2f} INR")
        print(f"  R2:   {eval_res['metrics']['r2_score']:.4f}")
        
        # Log to MLflow
        params = {k: str(v) for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))}
        log_regression_run(
            model_name=name,
            model=model,
            params=params,
            metrics=eval_res["metrics"],
            y_true=y_test_reg,
            y_pred=eval_res["y_pred"],
            feature_importance=eval_res["feature_importance"]
        )
        
        if eval_res["metrics"]["rmse"] < best_reg_rmse:
            best_reg_rmse = eval_res["metrics"]["rmse"]
            best_reg_name = name
            best_reg_model = model
            best_reg_feat_imp = eval_res["feature_importance"]
            
    print(f"\n[WINNER] Best Regression Model Selected: {best_reg_name} (RMSE: {best_reg_rmse:.2f} INR)")
    
    # ----------------------------------------------------
    # 8. Save Production Artifacts
    # ----------------------------------------------------
    print("\n[*] Persisting Production Artifacts to models/ ...")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(best_clf_model, MODELS_DIR / "best_classifier.joblib")
    joblib.dump(best_reg_model, MODELS_DIR / "best_regressor.joblib")
    
    if best_clf_feat_imp is not None:
        best_clf_feat_imp.to_csv(MODELS_DIR / "classifier_feature_importance.csv", index=False)
    if best_reg_feat_imp is not None:
        best_reg_feat_imp.to_csv(MODELS_DIR / "regressor_feature_importance.csv", index=False)
        
    summary = {
        "best_classifier": {
            "name": best_clf_name,
            "metrics": clf_results[best_clf_name]
        },
        "all_classifiers": clf_results,
        "best_regressor": {
            "name": best_reg_name,
            "metrics": reg_results[best_reg_name]
        },
        "all_regressors": reg_results,
        "feature_names": all_transformed_feature_names
    }
    
    with open(MODELS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("[SUCCESS] All artifacts saved successfully!")
    print("\nPipeline execution complete.")
    return summary


if __name__ == "__main__":
    run_pipeline()
