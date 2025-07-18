# Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Loading Model

model = joblib.load("fraud_rf_model.pk1")

# Configure Streamlit Page

st.set_page_config(page_title = "Fraud Detection", layout = "centered")

# Title and Description

st.title("💳 Fraud Detection Dashboard")
st.markdown("Enter transaction details below to predict if it's fraudulent")

# Layout: Input & Output columns
col1, col2 = st.columns([1, 1])

# Input Form
with col1:
    st.header("🔍 Transaction Input")

    with st.form("Prediction Form"):
        tx_type = st.selectbox('Transaction Type', ['CASH OUT', 'PAYMENT', 'TRANSFER', 'DEBIT'])
        amount = st.number_input("Amount", min_value = 0.0, step = 10.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value = 0.0, step = 10.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", min_value = 0.0, step = 10.0)
        newbalanceDest = st.number_input("New Balance (Destination)", min_value = 0.0, step = 10.0)
        
        # Use date_input and time_input instead of datetime_input
        date = st.date_input("Transaction Date", datetime(2024, 1, 1).date())
        time = st.time_input("Transaction Time", datetime(2024, 1, 1, 12, 0).time())
        user_datetime = datetime.combine(date, time)

        # Convert datetime to step (hour difference)
        start_datetime = datetime(2024, 1, 1)
        step = int((user_datetime - start_datetime).total_seconds() // 3600)
        step = max(1, min(step, 744))  # Clamp between 1 and 744


        submitted = st.form_submit_button("Predict")

# Helper function for dynamic color
def get_color(prob):
    if prob > 0.8:
        return 'darkred'
    elif prob > 0.5:
        return 'orange'
    else:
        return 'green'

# Gauge display with dynamic color
def show_gauge(probability):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(0, 1, color='#eee', height=0.3)
    ax.barh(0, probability, color=get_color(probability), height=0.3, edgecolor='black')
    ax.text(probability + 0.02, 0, f"{probability:.0%}", va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0%', '50%', '100%'])
    ax.set_title("Fraud Probability", fontsize=12)
    st.pyplot(fig)

# Process Input & Feature Engineering

if submitted:

    # Validation Checks
    if amount > oldbalanceOrg:
        st.warning("⚠️ Warning: Transaction amount exceeds origin account balance.")
    if newbalanceDest < oldbalanceDest:
        st.warning("⚠️ Warning: Destination balance decreased — unusual for this transaction.")

    input_data = pd.DataFrame([{
        'type': tx_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'step': step
    }])

# Feature Engineering

    input_data['day'] = input_data['step'] // 24
    input_data['log_amount'] = np.log1p(input_data['amount'])
    input_data['balance_delta_orig'] = input_data['oldbalanceOrg'] - input_data['amount']
    input_data['balance_delta_dest'] = input_data['newbalanceDest'] - input_data['oldbalanceDest']
    input_data['amount_to_balance_ratio'] = input_data['amount'] / (input_data['oldbalanceOrg']+1)

    input_data.drop(columns=['step'], inplace=True)

# Encode the 'type' column
    type_mapping = {'CASH OUT': 0, 'PAYMENT': 1, 'CASH IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
    input_data['type'] = input_data['type'] .str.upper().map(type_mapping)

# Match the column order used during model trainning

    feature_cols = ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'day',
        'log_amount', 'balance_delta_orig', 'balance_delta_dest', 'amount_to_balance_ratio']

    input_data = input_data[feature_cols]

# Make the predictions

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

# Display the results
    with col2:
        st.header('📊 Prediction Result')
        if pred == 1:
            st.error(f"🚨 **Prediction:** Fraudulent Transaction")
            st.progress(100)
        else:
            st.success("✅ **Prediction:** Transaction is Legitimate")
            st.balloons()
    
        st.metric(label = 'Fraud Probability', value = f"{prob: .2%}")
        show_gauge(prob)

    # Risk Level Indicator
        st.markdown("---")
        st.markdown("### 🧠 Risk Level")
        if prob > 0.8:
            st.warning("🟥 **High Risk**: This transaction is highly likely to be fraudulent.")
        elif prob > 0.5:
            st.info("🟧 **Medium Risk**: There's a moderate chance of fraud.")
        else:
            st.success("🟩 **Low Risk**: This transaction is likely safe.")
    # Gauge Color Legend
        st.markdown("### 📘 Gauge Color Legend")
        st.markdown("""
        - 🟩 **Low Risk**: 0% – 50%  
        - 🟧 **Medium Risk**: 50% – 80%  
        - 🟥 **High Risk**: 80% – 100%
        """)