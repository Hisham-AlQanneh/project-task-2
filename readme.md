# Credit Score Prediction System

## Overview
This project is a **Credit Score Prediction System** that uses multiple machine learning models to predict a user's credit score category (**Poor, Standard, Good**).

The system consists of:
- **Frontend** built with **Streamlit**
- **Backend API** built with **Flask**
- **Machine Learning Models** (Logistic Regression, XGBoost, Random Forest)

---

## Project Structure
```
project/
│── app.py              # Streamlit frontend
│── backend.py          # Flask backend API
│── requirements.txt    # Python dependencies
│── credit_score_model2.pkl
│── credit_score_modelxgb1.pkl
│── credit_score_rforest.pkl
│── val_results.csv     # Validation metrics (optional)
│── test_results.csv    # Test metrics (optional)
```

---

## Features
- Select between multiple ML models
- Input categorical and numerical credit features
- Real-time credit score prediction
- Visualization of training and testing metrics
- REST API for model inference

---

## Models Used
- Logistic Regression
- XGBoost Classifier
- Random Forest Classifier

Each model outputs probabilities for:
- Poor
- Standard
- Good

---

## Installation

### 1. Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start Backend (Flask API)
```bash
python backend.py
```
Backend will run at:
```
http://127.0.0.1:5000
```

### 2. Start Frontend (Streamlit)
```bash
streamlit run app.py
```
Frontend will open in your browser automatically.

---

## API Endpoint

### Predict Credit Score
**POST** `/predict/<model_name>`

**Model Names:**
- `logreg`
- `xgb`
- `rf`

**Request Body (JSON):**
```json
{
  "features": [
    1,
    0,
    0,
    1,
    25,
    50000,
    12,
    2000,
    0.45,
    300,
    1500
  ]
}
```

**Response:**
```json
{
  "Poor": 0.12,
  "Standard": 0.63,
  "Good": 0.25
}
```

---

## Training Results Visualization
If available, the app displays:
- Validation Accuracy & Loss
- Test Accuracy & Loss
- Epoch-wise performance graphs

Files required:
- `val_results.csv`
- `test_results.csv`

---

## Technologies Used
- Python
- Streamlit
- Flask
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib

---

## Notes
- Ensure trained model `.pkl` files exist before running backend
- Backend must be running before using the frontend
