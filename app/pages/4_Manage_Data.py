"""
EMIPredict AI - Financial Data Management (CRUD Administrative Interface)
Fulfills project requirements for complete CRUD operations on financial records.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st
import pandas as pd
from src.data_loader import load_dataset
from src.config import DATA_DIR

st.set_page_config(page_title="Manage Financial Data | EMIPredict AI", page_icon="📂", layout="wide")

st.title("📂 Financial Data Management (CRUD Interface)")
st.markdown("Administrative portal to create, read, update, and delete loan applicant profiles.")

# File persistence path for admin operations
ADMIN_DB_PATH = DATA_DIR / "admin_financial_records.csv"

def get_admin_data() -> pd.DataFrame:
    if ADMIN_DB_PATH.exists():
        return pd.read_csv(ADMIN_DB_PATH)
    else:
        df = load_dataset(use_synthetic_if_missing=True).head(50).copy()
        df.insert(0, "applicant_id", [f"APP-{1000 + i}" for i in range(len(df))])
        df.to_csv(ADMIN_DB_PATH, index=False)
        return df

def save_admin_data(df: pd.DataFrame):
    df.to_csv(ADMIN_DB_PATH, index=False)

df_admin = get_admin_data()

crud_tabs = st.tabs(["📖 Read & Search", "➕ Create Record", "✏️ Update Record", "🗑️ Delete Record"])

# 1. READ & SEARCH
with crud_tabs[0]:
    st.subheader("Search & Filter Financial Records")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_filter = st.multiselect("Filter by Scenario:", df_admin["emi_scenario"].unique())
    with c2:
        status_filter = st.multiselect("Filter by Eligibility:", df_admin["emi_eligibility"].unique())
    with c3:
        salary_range = st.slider("Filter by Monthly Salary (INR):", 
                                 float(df_admin["monthly_salary"].min()), 
                                 float(df_admin["monthly_salary"].max()), 
                                 (float(df_admin["monthly_salary"].min()), float(df_admin["monthly_salary"].max())))
        
    filtered_df = df_admin.copy()
    if scenario_filter:
        filtered_df = filtered_df[filtered_df["emi_scenario"].isin(scenario_filter)]
    if status_filter:
        filtered_df = filtered_df[filtered_df["emi_eligibility"].isin(status_filter)]
    filtered_df = filtered_df[(filtered_df["monthly_salary"] >= salary_range[0]) & (filtered_df["monthly_salary"] <= salary_range[1])]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df_admin)} records:**")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.download_button(
        "📥 Download Filtered Records as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_loan_applicants.csv",
        mime="text/csv"
    )

# 2. CREATE RECORD
with crud_tabs[1]:
    st.subheader("Create New Loan Applicant Profile")
    with st.form("create_applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_id = f"APP-{len(df_admin) + 1001}"
            st.text_input("Applicant ID", value=new_id, disabled=True)
            new_age = st.number_input("Age", min_value=21, max_value=70, value=30)
            new_gender = st.selectbox("Gender", ["Male", "Female"])
            new_salary = st.number_input("Monthly Salary (INR)", min_value=15000.0, value=75000.0, step=5000.0)
            new_scenario = st.selectbox("EMI Scenario", df_admin["emi_scenario"].unique())
        with col2:
            new_credit = st.slider("Credit Score", 300, 850, 720)
            new_emi = st.number_input("Current EMI Outflow (INR)", min_value=0.0, value=5000.0, step=1000.0)
            new_amount = st.number_input("Requested Loan Amount (INR)", min_value=10000.0, value=250000.0, step=10000.0)
            new_tenure = st.number_input("Requested Tenure (Months)", min_value=3, max_value=84, value=24)
            new_status = st.selectbox("Assigned Eligibility", ["Eligible", "High_Risk", "Not_Eligible"])
            
        submitted_create = st.form_submit_button("Add Record to System")
        if submitted_create:
            new_row = {
                "applicant_id": new_id,
                "age": new_age,
                "gender": new_gender,
                "monthly_salary": new_salary,
                "emi_scenario": new_scenario,
                "credit_score": new_credit,
                "current_emi_amount": new_emi,
                "requested_amount": new_amount,
                "requested_tenure": new_tenure,
                "emi_eligibility": new_status,
                "max_monthly_emi": round(new_salary * 0.35, 2)
            }
            # Fill other default columns if missing
            for col in df_admin.columns:
                if col not in new_row:
                    new_row[col] = df_admin[col].iloc[0]
            new_row_df = pd.DataFrame([new_row])
            df_admin = pd.concat([df_admin, new_row_df], ignore_index=True)
            save_admin_data(df_admin)
            st.success(f"Record {new_id} created successfully!")
            st.rerun()

# 3. UPDATE RECORD
with crud_tabs[2]:
    st.subheader("Update Existing Applicant Profile")
    record_to_update = st.selectbox("Select Applicant ID to modify:", df_admin["applicant_id"].unique())
    current_rec = df_admin[df_admin["applicant_id"] == record_to_update].iloc[0]
    
    with st.form("update_applicant_form"):
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            up_salary = st.number_input("Monthly Salary (INR)", min_value=10000.0, value=float(current_rec["monthly_salary"]), step=5000.0)
            up_credit = st.slider("Credit Score", 300, 850, int(current_rec["credit_score"]))
        with u_col2:
            up_current_emi = st.number_input("Current EMI (INR)", min_value=0.0, value=float(current_rec["current_emi_amount"]), step=1000.0)
            up_status = st.selectbox("Eligibility Status", ["Eligible", "High_Risk", "Not_Eligible"], 
                                     index=["Eligible", "High_Risk", "Not_Eligible"].index(current_rec["emi_eligibility"]))
            
        submitted_update = st.form_submit_button("Update Applicant Record")
        if submitted_update:
            idx = df_admin[df_admin["applicant_id"] == record_to_update].index[0]
            df_admin.at[idx, "monthly_salary"] = up_salary
            df_admin.at[idx, "credit_score"] = up_credit
            df_admin.at[idx, "current_emi_amount"] = up_current_emi
            df_admin.at[idx, "emi_eligibility"] = up_status
            save_admin_data(df_admin)
            st.success(f"Applicant {record_to_update} successfully updated!")
            st.rerun()

# 4. DELETE RECORD
with crud_tabs[3]:
    st.subheader("Delete Applicant Record")
    record_to_delete = st.selectbox("Select Applicant ID to remove:", df_admin["applicant_id"].unique(), key="del_select")
    st.warning(f"Are you sure you want to permanently delete record: **{record_to_delete}**?")
    if st.button("🚨 Confirm Permanent Deletion"):
        df_admin = df_admin[df_admin["applicant_id"] != record_to_delete].reset_index(drop=True)
        save_admin_data(df_admin)
        st.success(f"Record {record_to_delete} deleted successfully!")
        st.rerun()
