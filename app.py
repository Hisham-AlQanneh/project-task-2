import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Score App", layout="wide")

page = st.sidebar.selectbox("Choose Page", ["Prediction", "Training Results"])

if page == "Prediction":
    st.title("Credit Score Prediction Demo")

    model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "XGBoost", "Random Forest"]
    )

    model_map = {
        "Logistic Regression": "logreg",
        "XGBoost": "xgb",
        "Random Forest": "rf"
    }

    selected_model = model_map[model_name]

    st.subheader("Categorical Features")
    credit_mix_good = st.checkbox("Credit Mix: Good")
    credit_mix_standard = st.checkbox("Credit Mix: Standard")
    pay_min_no = st.checkbox("Payment of Min Amount: No")
    pay_min_yes = st.checkbox("Payment of Min Amount: Yes")

    st.subheader("Numerical Features")
    age = st.number_input("Age", value=0.0)
    annual_income = st.number_input("Annual Income", value=0.0)
    interest_rate = st.number_input("Interest Rate", value=0.0)
    outstanding_debt = st.number_input("Outstanding Debt", value=0.0)
    credit_util = st.number_input("Credit Utilization Ratio", value=0.0)
    total_emi = st.number_input("Total EMI per Month", value=0.0)
    monthly_balance = st.number_input("Monthly Balance", value=0.0)

    features = [
        int(credit_mix_good),
        int(credit_mix_standard),
        int(pay_min_no),
        int(pay_min_yes),
        age,
        annual_income,
        interest_rate,
        outstanding_debt,
        credit_util,
        total_emi,
        monthly_balance
    ]

    if st.button("Predict Credit Score"):
        response = requests.post(
            f"http://127.0.0.1:5000/predict/{selected_model}",
            json={"features": features}
        )

        if response.status_code == 200:
            st.success(f"Prediction using {model_name}")
            st.json(response.json())
        else:
            st.error("Backend error")

elif page == "Training Results":
    st.title("Model Training Results")

    st.write("This page shows the validation and test metrics for your trained models.")

    try:
        val_results = pd.read_csv("val_results.csv")
        test_results = pd.read_csv("test_results.csv")

        st.subheader("Validation Metrics")
        st.dataframe(val_results)

        st.subheader("Test Metrics")
        st.dataframe(test_results)

        st.subheader("Metrics Visualization")
        fig, ax = plt.subplots()
        ax.plot(val_results['epoch'], val_results['accuracy'], label="Validation Accuracy")
        ax.plot(test_results['epoch'], test_results['accuracy'], label="Test Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy over Epochs")
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(val_results['epoch'], val_results['loss'], label="Validation Loss")
        ax2.plot(test_results['epoch'], test_results['loss'], label="Test Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss over Epochs")
        ax2.legend()
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("Training results CSV files not found. Make sure 'val_results.csv' and 'test_results.csv' exist in the app folder.")
