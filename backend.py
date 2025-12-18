print("BACKEND FILE STARTED")


from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import pandas as pd


best_logreg = joblib.load("credit_score_model2.pkl")
best_xgb = joblib.load("credit_score_modelxgb1.pkl")
best_rf = joblib.load("credit_score_rforest.pkl")


app = Flask(__name__)
CORS(app)  


@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    try:
        features = request.json["features"]

        columns = [
            "Credit_Mix_Good",
            "Credit_Mix_Standard",
            "Payment_of_Min_Amount_No",
            "Payment_of_Min_Amount_Yes",
            "Age",
            "Annual_Income",
            "Interest_Rate",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Monthly_Balance"
        ]

        X = pd.DataFrame([features], columns=columns)

        classes = ["Poor", "Standard", "Good"]

        if model_name in ["logreg", "best_logreg"]:
            # ✅ KERAS MODEL
            probs = best_logreg.predict(X)[0]

        elif model_name in ["xgb", "best_xgb"]:
            probs = best_xgb.predict_proba(X)[0]

        elif model_name in ["rf", "best_rf"]:
            probs = best_rf.predict_proba(X)[0]

        else:
            return jsonify({"error": "Model not found"}), 400

        result = dict(zip(classes, map(float, probs)))
        return jsonify(result)

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)   # VERY IMPORTANT
        return jsonify({"error": str(e)}), 500



@app.route("/")
def home():
    return "Credit Score API is running"

if __name__ == "__main__":
    app.run(debug=True)
