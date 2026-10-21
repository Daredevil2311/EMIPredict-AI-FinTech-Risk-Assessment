"""
EMIPredict AI - Interactive Exploratory Data Analysis (EDA)
Addresses the 15 Core EDA Questions for FinTech Lending Risk.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.feature_engineering import compute_financial_features

st.set_page_config(page_title="EDA & Insights | EMIPredict AI", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis & FinTech Insights")
st.markdown("Investigating financial capacity distributions, scenario patterns, and statistical significance across 400,000 borrower profiles.")

@st.cache_data
def get_eda_data():
    raw = load_dataset(use_synthetic_if_missing=True)
    return compute_financial_features(raw)

df = get_eda_data()

tabs = st.tabs([
    "1. Overview & Data Quality",
    "2. Scenario & Demographic Patterns",
    "3. Risk & Credit Correlations",
    "4. Statistical Hypothesis Tests"
])

# TAB 1: Data Characteristics & Quality
with tabs[0]:
    st.subheader("📋 Dataset Characteristics (EDA Questions 1, 2, 5, 6)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Input Variables", f"{len(df.columns)}")
    c3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    c4.metric("Memory Usage", f"{df.memory_usage().sum() / (1024**2):.2f} MB")
    
    st.markdown("#### Sample Records (Raw Data)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("#### Numerical Summary Statistics (EDA Questions 10 & 11)")
    st.dataframe(df.describe().T[["mean", "std", "min", "50%", "max"]].round(2), use_container_width=True)

# TAB 2: Scenario Patterns
with tabs[1]:
    st.subheader("🏦 Lending Scenarios & Demographics (EDA Questions 3, 13)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Eligibility Distribution Across Scenarios**")
        scenario_ct = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8, 4.5))
        scenario_ct.plot(kind="bar", stacked=True, color=["#10B981", "#F59E0B", "#EF4444"], ax=ax)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.set_title("EMI Eligibility Breakdown by Lending Scenario")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.markdown("**Requested Loan Amount by Scenario**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=df, x="emi_scenario", y="requested_amount", palette="Blues_r", ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel("Requested Amount (Log Scale INR)")
        ax.set_xlabel("")
        ax.set_title("Loan Amount Distribution Across Scenarios")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# TAB 3: Risk & Correlation
with tabs[2]:
    st.subheader("🎯 Risk Drivers & Correlation Matrix (EDA Questions 4, 7)")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Credit Score vs Eligibility Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df, x="credit_score", hue="emi_eligibility", 
                    palette={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
                    common_norm=False, fill=True, alpha=0.4, ax=ax)
        ax.set_title("Credit Score Density Across Eligibility Status")
        ax.set_xlabel("Credit Score (300-850)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_b:
        st.markdown("**Monthly Salary vs Maximum Safe EMI Capacity**")
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(sample_df["monthly_salary"], sample_df["max_monthly_emi"], 
                             c=sample_df["credit_score"], cmap="viridis", alpha=0.6, edgecolors="none")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Credit Score")
        ax.set_xlabel("Monthly Salary (INR)")
        ax.set_ylabel("Max Monthly EMI (INR)")
        ax.set_title("Capacity Scaling: Salary & Credit Score vs Safe EMI")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Correlation Matrix with Target Variable (max_monthly_emi)")
    num_cols = ["monthly_salary", "credit_score", "disposable_income", "total_monthly_expenses", 
                "current_emi_amount", "bank_balance", "emergency_fund", "financial_health_score", "max_monthly_emi"]
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Pearson Correlation Heatmap of Key Financial Predictors")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# TAB 4: Statistical Hypothesis Tests
with tabs[3]:
    st.subheader("🔬 Statistical Hypothesis Testing (Capstone Guideline Requirements)")
    st.markdown(r"""
    To validate feature significance beyond empirical correlations, formal hypothesis tests were executed:
    1. **Chi-Square Test ($\chi^2$) of Independence**: Categorical predictors vs. EMI Eligibility Target.
    2. **One-Way ANOVA ($F$-test)**: Continuous features across 3 eligibility classes.
    3. **Pearson & Spearman Rank Correlation**: Linear & Monotonic association with `max_monthly_emi`.
    """)
    
    rep_dir = root_dir / "docs"
    chi2_path = rep_dir / "chi_square_tests.csv"
    anova_path = rep_dir / "anova_tests.csv"
    corr_path = rep_dir / "correlation_tests.csv"
    
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.markdown("**Chi-Square Independence Tests:**")
        if chi2_path.exists():
            st.dataframe(pd.read_csv(chi2_path), hide_index=True, use_container_width=True)
        else:
            st.info("Run training pipeline to generate statistical tests.")
            
    with t_c2:
        st.markdown("**One-Way ANOVA Tests:**")
        if anova_path.exists():
            st.dataframe(pd.read_csv(anova_path), hide_index=True, use_container_width=True)
        else:
            st.info("Run training pipeline to generate statistical tests.")
            
    st.markdown("**Correlation Battery with Target Variable:**")
    if corr_path.exists():
        st.dataframe(pd.read_csv(corr_path), hide_index=True, use_container_width=True)
