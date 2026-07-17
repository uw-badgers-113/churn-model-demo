"""
Assignment 2 — Part 2
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Healthy Meals Churn Predictor",
                   page_icon="🍽️", layout="centered")


@st.cache_resource
def load_pipeline():
    with open("churn_pipeline_healthy_meals.pkl", "rb") as f:
        return pickle.load(f)


pipe = load_pipeline()

INCOME_CHOICES    = ["Low", "Medium", "High", "Very High"]
EDUCATION_CHOICES = ["High School", "Graduate", "Post-Graduate", "Other"]
DEVICE_CHOICES    = ["Desktop-only", "Mobile-only", "Multi-device"]

st.title("Healthy Meals — Churn Predictor")
st.caption("Enter a customer's prior-year activity and demographics to estimate churn risk.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Activity (prior year)")
    tot_num_sessions     = st.number_input("Total # sessions", min_value=0, value=40, step=1)
    gross_session_length = st.number_input("Gross session length (min)", min_value=0, value=1800, step=10)
    active_days          = st.number_input("Active days", min_value=0, value=30, step=1)
    active_quarters      = st.slider("Active quarters", 0, 4, 4)
with col2:
    st.subheader("Demographics")
    age                = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
    tech_comfort_score = st.slider("Tech comfort score", 1, 10, 7)
    income_level       = st.selectbox("Income level", INCOME_CHOICES, index=2)
    education          = st.selectbox("Education", EDUCATION_CHOICES, index=1)
    device_type        = st.selectbox("Device type", DEVICE_CHOICES, index=2)

if st.button("Predict churn", type="primary"):
    # derive avg sessions / active quarter (same null / divide-by-zero-safe rule as the SQL)
    avg_spq = tot_num_sessions / active_quarters if active_quarters else 0.0

    row = pd.DataFrame([{
        "TOT_NUM_SESSIONS": tot_num_sessions,
        "GROSS_SESSION_LENGTH": gross_session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_spq,
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score,
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type,
    }])

    p_renew = float(pipe.predict_proba(row)[:, 1][0])
    p_churn = 1.0 - p_renew

    st.divider()
    st.metric("Churn probability", f"{p_churn:.1%}")
    st.progress(min(max(p_churn, 0.0), 1.0))
    if p_churn >= 0.5:
        st.error("HIGH churn risk")
    elif p_churn >= 0.3:
        st.warning("MEDIUM churn risk")
    else:
        st.success("LOW churn risk")
    st.caption(f"Renewal probability: {p_renew:.1%}")
