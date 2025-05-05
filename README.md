# 🛍️ Demand Forecasting App

This project is an end-to-end machine learning application that analyzes retail sales data to forecast future product demand. It covers data analysis, model training, prediction via Flask, and an interactive Streamlit interface.

## 🔍 Project Overview

### 1. 📊 Exploratory Data Analysis (EDA)
- Examined sales trends and distributions
- Handled missing and outlier values
- Feature engineering with time-based variables (lag, rolling std, etc.)

### 2. 🤖 Machine Learning Model
- **Model Used:** XGBoost Regressor
- **Goal:** Predict daily item-level sales
- **Features:**
  - `store`, `item`, `year`, `month`, `dayofweek`, `is_weekend`
  - `lag_7`, `lag_30`, `rolling_std_7`, `sales_diff`

### 3. 🔁 Flask API
- A lightweight REST API was developed to serve the ML model.
- `/predict` endpoint returns predictions based on input features.

### 4. 🎨 Streamlit Web App
- A user-friendly interface built with Streamlit.
- Users can interact with the model and see visualized predictions and SHAP values.

## 🛠️ Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Plotly
- XGBoost
- Flask
- Streamlit
- SHAP

## 🚀 How to Run the Project
streamlit link : http://localhost:8501/

### 1. Train the Model:
```bash
python train_model.py
