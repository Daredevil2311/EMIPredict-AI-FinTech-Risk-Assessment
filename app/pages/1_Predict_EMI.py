"""
EMIPredict AI - Real-Time EMI Prediction & Risk Assessment Tool
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np

from src.predict import EMIPredictor

st.set_page_config(page_title="Dual EMI Prediction | EMIPredict AI", page_icon="⚡", layout="wide")

# Modern Styling
st.markdown("""
<style>
    .result-box-eligible {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .result-box-highrisk {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
    }
    .result-box-noteligible {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    .metric-badge {
        font-family: monospace;
        font-weight: 700;
        font-size: 1.8rem;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Real-Time EMI Risk Assessment & Dual Prediction")
st.markdown("Evaluate borrower creditworthiness and calculate exact safe maximum monthly EMI capacity using the production-trained models.")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return EMIPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Failed to load trained models: {e}. Please ensure pipeline has been executed.")
    st.stop()

# Presets for Live Evaluation Demonstration
st.subheader("💡 Quick Evaluation Presets")
preset_choice = st.selectbox(
    "Load sample applicant profile for testing:",
    [
        "Custom Input",
        "Profile 1: Prime Salaried Professional (Eligible)",
        "Profile 2: Marginal Subprime Applicant (High_Risk)",
        "Profile 3: Over-leveraged Small Business (Not_Eligible)"
    ]
)

# Preset values dictionary
presets = {
    "Profile 1: Prime Salaried Professional (Eligible)": {
        "age": 34, "gender": "Male", "marital_status": "Married", "education": "Post Graduate",
        "monthly_salary": 95000.0, "employment_type": "Private", "years_of_employment": 8.0,
        "company_type": "MNC", "house_type": "Own", "monthly_rent": 0.0, "family_size": 3,
        "dependents": 1, "school_fees": 5000.0, "college_fees": 0.0, "travel_expenses": 4000.0,
        "groceries_utilities": 14000.0, "other_monthly_expenses": 5000.0, "existing_loans": "Yes",
        "current_emi_amount": 8000.0, "credit_score": 780, "bank_balance": 350000.0,
        "emergency_fund": 200000.0, "emi_scenario": "Vehicle EMI", "requested_amount": 500000.0,
        "requested_tenure": 48
    },
    "Profile 2: Marginal Subprime Applicant (High_Risk)": {
        "age": 28, "gender": "Female", "marital_status": "Single", "education": "Graduate",
        "monthly_salary": 42000.0, "employment_type": "Private", "years_of_employment": 3.0,
        "company_type": "Startup", "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 2,
        "dependents": 1, "school_fees": 2000.0, "college_fees": 0.0, "travel_expenses": 3000.0,
        "groceries_utilities": 9000.0, "other_monthly_expenses": 3000.0, "existing_loans": "Yes",
        "current_emi_amount": 7500.0, "credit_score": 620, "bank_balance": 45000.0,
        "emergency_fund": 20000.0, "emi_scenario": "Personal Loan EMI", "requested_amount": 250000.0,
        "requested_tenure": 36
    },
    "Profile 3: Over-leveraged Small Business (Not_Eligible)": {
        "age": 45, "gender": "Male", "marital_status": "Married", "education": "High School",
        "monthly_salary": 30000.0, "employment_type": "Self-employed", "years_of_employment": 5.0,
        "company_type": "Startup", "house_type": "Rented", "monthly_rent": 10000.0, "family_size": 5,
        "dependents": 3, "school_fees": 5000.0, "college_fees": 3000.0, "travel_expenses": 2500.0,
        "groceries_utilities": 11000.0, "other_monthly_expenses": 4000.0, "existing_loans": "Yes",
        "current_emi_amount": 9000.0, "credit_score": 510, "bank_balance": 8000.0,
        "emergency_fund": 0.0, "emi_scenario": "Personal Loan EMI", "requested_amount": 400000.0,
        "requested_tenure": 24
    }
}

p = presets.get(preset_choice, {})

with st.form("emi_prediction_form"):
    st.markdown("### 📋 Applicant Financial Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal & Demographics**")
        age = st.slider("Age", 21, 65, int(p.get("age", 30)))
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if p.get("gender", "Male") == "Male" else 1)
        marital_status = st.selectbox("Marital Status", ["Single", "Married"], index=0 if p.get("marital_status", "Single") == "Single" else 1)
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"], 
                                 index=["High School", "Graduate", "Post Graduate", "Professional"].index(p.get("education", "Graduate")))
        
        st.markdown("**Housing & Dependents**")
        house_type = st.selectbox("House Ownership", ["Rented", "Own", "Family"],
                                  index=["Rented", "Own", "Family"].index(p.get("house_type", "Rented")))
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, value=float(p.get("monthly_rent", 12000.0)), step=1000.0)
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=int(p.get("family_size", 3)))
        dependents = st.number_input("Dependents", min_value=0, max_value=8, value=int(p.get("dependents", 1)))
        
    with col2:
        st.markdown("**Employment & Income**")
        monthly_salary = st.number_input("Monthly Gross Salary (₹)", min_value=10000.0, max_value=500000.0, 
                                          value=float(p.get("monthly_salary", 65000.0)), step=5000.0)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"],
                                       index=["Private", "Government", "Self-employed"].index(p.get("employment_type", "Private")))
        years_of_employment = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, 
                                              value=float(p.get("years_of_employment", 5.0)), step=0.5)
        company_type = st.selectbox("Company Type", ["Startup", "Mid-sized", "Large Enterprise", "MNC", "Government"],
                                    index=["Startup", "Mid-sized", "Large Enterprise", "MNC", "Government"].index(p.get("company_type", "Mid-sized")))
        
        st.markdown("**Monthly Living Obligations**")
        school_fees = st.number_input("School Fees (₹)", min_value=0.0, value=float(p.get("school_fees", 2000.0)), step=500.0)
        college_fees = st.number_input("College Fees (₹)", min_value=0.0, value=float(p.get("college_fees", 0.0)), step=1000.0)
        travel_expenses = st.number_input("Travel / Commute (₹)", min_value=0.0, value=float(p.get("travel_expenses", 3000.0)), step=500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=1000.0, value=float(p.get("groceries_utilities", 12000.0)), step=1000.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, value=float(p.get("other_monthly_expenses", 3000.0)), step=500.0)

    with col3:
        st.markdown("**Existing Debts & Liquidity**")
        existing_loans = st.selectbox("Existing Active Loans?", ["Yes", "No"], 
                                      index=0 if p.get("existing_loans", "No") == "Yes" else 1)
        current_emi_amount = st.number_input("Current Monthly EMI Outflow (₹)", min_value=0.0, 
                                              value=float(p.get("current_emi_amount", 5000.0)), step=1000.0)
        credit_score = st.slider("Credit Bureau Score (CIBIL/Experian)", 300, 850, int(p.get("credit_score", 720)))
        bank_balance = st.number_input("Current Bank Balance (₹)", min_value=0.0, 
                                       value=float(p.get("bank_balance", 120000.0)), step=10000.0)
        emergency_fund = st.number_input("Emergency Reserve Fund (₹)", min_value=0.0, 
                                         value=float(p.get("emergency_fund", 50000.0)), step=10000.0)
        
        st.markdown("**Loan Application Details**")
        emi_scenario = st.selectbox("EMI Scenario", [
            "E-commerce Shopping EMI",
            "Home Appliances EMI",
            "Vehicle EMI",
            "Personal Loan EMI",
            "Education EMI"
        ], index=["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"].index(p.get("emi_scenario", "Personal Loan EMI")))
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=5000.0, max_value=2000000.0, 
                                           value=float(p.get("requested_amount", 200000.0)), step=10000.0)
        requested_tenure = st.slider("Requested Repayment Tenure (Months)", 3, 84, int(p.get("requested_tenure", 24)))

    submit_button = st.form_submit_button("🚀 Run Dual Risk Assessment & EMI Calculation", use_container_width=True)

if submit_button:
    applicant_dict = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }
    
    res = predictor.predict_single(applicant_dict)
    
    st.markdown("---")
    st.subheader("🎯 Assessment Results")
    
    res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
    
    with res_col1:
        eligibility = res["eligibility"]
        if eligibility == "Eligible":
            st.success("### ✅ ELIGIBLE")
            st.caption("Low Default Risk • Prime Borrower")
        elif eligibility == "High_Risk":
            st.warning("### ⚠️ HIGH RISK")
            st.caption("Marginal Affordability • Requires Mitigants")
        else:
            st.error("### 🛑 NOT ELIGIBLE")
            st.caption("Excessive Leverage • High Probability of Default")
            
        st.markdown("**Eligibility Probabilities:**")
        prob_df = pd.DataFrame(list(res["probabilities"].items()), columns=["Status", "Probability (%)"])
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
    with res_col2:
        st.info(f"### ₹{res['max_safe_monthly_emi']:,.2f}")
        st.caption("Maximum Safe Monthly EMI Capacity")
        
        st.metric("Financial Health Index", f"{res['financial_health_score']} / 100")
        st.metric("Net Free Cash Flow", f"₹{res['disposable_income']:,.2f}/mo")

    with res_col3:
        st.markdown("**Key Underwriting Ratios:**")
        st.write(f"- **Debt-to-Income (DTI):** `{res['debt_to_income_pct']}%` (Benchmark: < 40%)")
        st.write(f"- **Expense-to-Income (ETI):** `{res['expense_to_income_pct']}%`")
        st.write(f"- **Requested Monthly Burden:** `₹{res['requested_monthly_burden']:,.2f}/mo`")
        
        if res["risk_flags"]:
            st.markdown("**⚠️ Identified Risk Flags:**")
            for flag in res["risk_flags"]:
                st.markdown(f"- {flag}")
        else:
            st.markdown("✅ *No active risk flags detected.*")
            
    # Decision Card
    st.markdown("### 📝 AI Underwriter Final Decision")
    if eligibility == "Eligible":
        st.success(res["recommendation"])
    elif eligibility == "High_Risk":
        st.warning(res["recommendation"])
    else:
        st.error(res["recommendation"])
