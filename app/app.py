import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load trained model
model = joblib.load("model/sme_loan_model.pkl")

# ✅ Get the exact feature names used during training
expected_features = list(model.feature_names_in_)

# ✅ SME sectors (must match training data)
sectors = ["Retail", "Agriculture", "Transport (Boda Boda)", 
           "Tech Startup", "Hospitality", "Manufacturing", "Services"]

# ✅ App Title & Header
st.set_page_config(page_title="Kenyan SME Loan Approval Predictor", layout="wide")
st.title("💼 Kenyan SME Loan Approval Predictor")
st.write("Enter your SME details to check **Loan Approval Probability** in Kenyan Shillings (KES)")

# --- Sidebar: Input Fields ---
st.sidebar.header("📋 SME Business Details")
business_age = st.sidebar.slider("Business Age (years)", 1, 15, 3)
sector = st.sidebar.selectbox("Business Sector", sectors)
monthly_revenue = st.sidebar.number_input("Monthly Revenue (KES)", min_value=20000, max_value=2000000, value=500000, step=5000)
monthly_expenses = st.sidebar.number_input("Monthly Expenses (KES)", min_value=10000, max_value=1500000, value=200000, step=5000)
num_employees = st.sidebar.slider("Number of Employees", 1, 50, 5)
collateral_value = st.sidebar.number_input("Collateral Value (KES)", min_value=0, max_value=1000000, value=200000, step=10000)
requested_loan = st.sidebar.number_input("Requested Loan Amount (KES)", min_value=50000, max_value=2000000, value=300000, step=50000)
credit_history_score = st.sidebar.slider("Credit History Score (300–850)", 300, 850, 600)

# --- Convert inputs into DataFrame (initial)
raw_input = pd.DataFrame({
    "business_age_yrs": [business_age],
    "monthly_revenue_ksh": [monthly_revenue],
    "monthly_expenses_ksh": [monthly_expenses],
    "num_employees": [num_employees],
    "collateral_value_ksh": [collateral_value],
    "requested_loan_ksh": [requested_loan],
    "credit_history_score": [credit_history_score],
    # One-hot encode sectors like in training
    **{f"sector_{s}": [1 if s == sector else 0] for s in sectors[1:]}  # drop_first=True was used
})

# ✅ Align input with expected features from model training
aligned_input = pd.DataFrame(columns=expected_features)
aligned_input.loc[0] = 0  # fill with zeros
for col in raw_input.columns:
    if col in aligned_input.columns:
        aligned_input[col] = raw_input[col]

st.write("### SME Input Summary")
st.dataframe(aligned_input)

# --- Predict Approval ---
if st.button("🔍 Predict Loan Approval"):
    # ✅ Now aligned_input matches model’s expected columns
    pred = model.predict_proba(aligned_input)[0][1]  # probability of approval
    approved_prob = round(pred * 100, 2)
    
    if approved_prob >= 70:
        st.success(f"✅ **High Chance of Approval:** {approved_prob}%")
    elif approved_prob >= 50:
        st.warning(f"⚠️ **Moderate Chance of Approval:** {approved_prob}%")
    else:
        st.error(f"❌ **Low Chance of Approval:** {approved_prob}%")
    
    st.write("💡 **Recommendation:**")
    if approved_prob < 50:
        st.write("- Increase collateral value or monthly revenue.")
        st.write("- Improve credit history score before applying.")
    elif approved_prob < 70:
        st.write("- Slightly improve cash flow & reduce expenses.")
    else:
        st.write("- Your business is strong for loan approval.")

    # --- Feature importance chart ---
    st.subheader("📊 What Matters Most for Loan Approval")
    feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)
    top_features = feature_importances.head(8)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="Blues_r", ax=ax)
    ax.set_title("Top Features Influencing Loan Approval")
    st.pyplot(fig)

st.markdown("---")
st.caption("💡 Built with **Streamlit + RandomForest ML Model** for Kenyan SME loan predictions.")
