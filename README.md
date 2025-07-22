
# 💳 Fraud Detection Dashboard

A real-time interactive web application powered by machine learning to predict the likelihood of a financial transaction being fraudulent. This app is built using **Streamlit** and a trained **Random Forest classifier**, providing a clean user interface for analysts, engineers, or financial institutions to evaluate transaction risk.

---

## 🧠 Project Overview

Financial fraud poses a serious challenge across global payment systems. This dashboard provides an intelligent way to:

- Analyze individual transactions
- Predict the likelihood of fraud
- Visualize risk dynamically
- Provide immediate warnings for suspicious patterns

The model has been trained on transaction data including account balances, amount transferred, transaction type, and engineered features like log-scaled amounts and balance changes.

---

## 📊 App Features

✅ **Transaction Input Interface**  
Users can enter transaction details:
- Sender & Receiver balances before/after the transaction
- Transaction amount and type
- Timestamp via calendar + clock picker

📈 **Fraud Probability Gauge**  
A real-time horizontal gauge displays the model's confidence visually, color-coded by risk:
- 🟩 Low (0–50%)
- 🟧 Medium (50–80%)
- 🟥 High (80–100%)

🧠 **Risk Level Interpretation**  
In addition to the numerical output, the app classifies and explains the level of risk with descriptive feedback.

⚠️ **Pre-check Warnings**  
Built-in checks highlight potentially abnormal transaction behavior (e.g. amount > sender balance, decreased destination balance).

🎯 **Model-backed Predictions**  
Uses a trained Random Forest model to classify fraud based on features including:
- `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, `newbalanceDest`, `step`
- Engineered features: log amounts, balance deltas, ratios

---

## 🔧 Tech Stack

| Tool/Library   | Purpose                        |
|----------------|-------------------------------|
| **Streamlit**  | Web app framework             |
| **Pandas**     | Data processing               |
| **NumPy**      | Numerical operations          |
| **Matplotlib** | Probability gauge visualization |
| **Joblib**     | Model serialization           |
| **scikit-learn** | Random Forest training       |

---

## 🏗️ Project Structure

```
fraud-detection-app/
│
├── FraudDetectionApp.py         # Streamlit frontend + prediction logic
├── fraud_rf_model.pk1           # Trained Random Forest classifier (binary)
├── requirements.txt             # All project dependencies
├── PS_20174392719_1491204439457.csv (optional)  # Training dataset (excluded from GitHub)
└── README.md                    # This file
```

---

## 🧪 Example Use Case

**Scenario:**  
A bank employee receives a request for a large transfer.

**Action:**  
They enter the details into the app:
- Type: `TRANSFER`
- Amount: `10,000`
- Origin balance before: `10,000`
- Destination balance before: `500`, after: `10,500`
- Time of day: `Jan 2, 2024 — 9:30 AM`

**Output:**
- Model predicts **High Risk**
- Probability: `81.0%`
- Gauge: 🟥
- Message: "🚨 Fraudulent Transaction Likely"

---

## 📌 Notes

- The app restricts transaction time to the same scale used during model training (steps 1–744, where 1 step = 1 hour).
- The dataset used for training was a synthetic but realistic bank transaction dataset simulating fraud patterns.
- File `PS_20174392719_1491204439457.csv` is **not included** in the repo due to size.

---

## 🙋‍♂️ Author

**Daniel Knuth**  
📫 [GitHub Profile](https://github.com/DanielKnuth)  
💡 Project guided by fraud detection research & hands-on ML modeling
