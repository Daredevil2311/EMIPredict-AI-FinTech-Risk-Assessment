"""
EMIPredict AI - MLflow Experiment Tracking & Model Registry Utility
Tracks runs, parameters, metrics, artifacts, and manages model registration.
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACTS_DIR, MODELS_DIR

# Set modern SQLite database tracking URI with local artifact directory
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "EMIPredict_Financial_Risk_Platform"


def setup_mlflow_experiment(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Creates or retrieves the MLflow experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = client.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    else:
        exp_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_classification_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    confusion_mat: np.ndarray,
    class_names: list,
    feature_importance: Optional[pd.DataFrame] = None
) -> str:
    """Logs a classification training run with parameters, metrics, and visualization artifacts."""
    setup_mlflow_experiment()
    
    with mlflow.start_run(run_name=f"Clf_{model_name}") as run:
        # 1. Tags & Parameters
        mlflow.set_tag("task_type", "classification")
        mlflow.set_tag("model_family", model_name)
        mlflow.log_params(params)
        
        # 2. Metrics
        mlflow.log_metrics(metrics)
        
        # 3. Artifact: Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        cm_path = MODELS_DIR / f"{model_name}_confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        
        # 4. Artifact: Feature Importance Plot
        if feature_importance is not None and not feature_importance.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_feats = feature_importance.head(15)
            sns.barplot(data=top_feats, x="importance", y="feature", hue="feature", legend=False, palette="viridis", ax=ax)
            ax.set_title(f"Top 15 Feature Importances - {model_name}")
            plt.tight_layout()
            feat_path = MODELS_DIR / f"{model_name}_feature_importance.png"
            fig.savefig(feat_path)
            plt.close(fig)
            mlflow.log_artifact(str(feat_path), artifact_path="plots")
            
        # 5. Log Model
        try:
            mlflow.sklearn.log_model(
                model, 
                name="model",
                serialization_format="pickle"
            )
        except Exception:
            # Fallback for models with custom estimators
            model_joblib = MODELS_DIR / f"{model_name}.joblib"
            joblib.dump(model, model_joblib)
            mlflow.log_artifact(str(model_joblib), artifact_path="model")
        return run.info.run_id


def log_regression_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: Optional[pd.DataFrame] = None
) -> str:
    """Logs a regression training run with evaluation metrics, scatter plot, and model artifacts."""
    setup_mlflow_experiment()
    
    with mlflow.start_run(run_name=f"Reg_{model_name}") as run:
        mlflow.set_tag("task_type", "regression")
        mlflow.set_tag("model_family", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Artifact: Actual vs Predicted Scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        sample_idx = np.random.choice(len(y_true), size=min(1000, len(y_true)), replace=False)
        ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, color="#1f77b4", edgecolors="none")
        max_val = max(np.max(y_true[sample_idx]), np.max(y_pred[sample_idx]))
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
        ax.set_title(f"Actual vs Predicted Max EMI - {model_name}")
        ax.set_xlabel("Actual Max Monthly EMI (INR)")
        ax.set_ylabel("Predicted Max Monthly EMI (INR)")
        ax.legend()
        plt.tight_layout()
        scatter_path = MODELS_DIR / f"{model_name}_actual_vs_predicted.png"
        fig.savefig(scatter_path)
        plt.close(fig)
        mlflow.log_artifact(str(scatter_path), artifact_path="plots")
        
        # Log Model
        try:
            mlflow.sklearn.log_model(
                model, 
                name="model",
                serialization_format="pickle"
            )
        except Exception:
            model_joblib = MODELS_DIR / f"{model_name}.joblib"
            joblib.dump(model, model_joblib)
            mlflow.log_artifact(str(model_joblib), artifact_path="model")
        return run.info.run_id


def get_experiment_leaderboard() -> pd.DataFrame:
    """Retrieves all tracked experiment runs as a clean pandas DataFrame for the Streamlit dashboard."""
    setup_mlflow_experiment()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
        
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    records = []
    for r in runs:
        row = {
            "Run ID": r.info.run_id[:8],
            "Run Name": r.data.tags.get("mlflow.runName", "Unnamed"),
            "Task Type": r.data.tags.get("task_type", "Unknown"),
            "Model": r.data.tags.get("model_family", "Unknown"),
            "Status": r.info.status,
            "Start Time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
        }
        # Add metrics
        for k, v in r.data.metrics.items():
            row[k] = round(v, 4)
        records.append(row)
        
    return pd.DataFrame(records)
