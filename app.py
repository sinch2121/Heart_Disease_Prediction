# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.pkl')

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    return df

def plot_probability(prediction_proba):
    st.write("### Prediction Probability")
    st.bar_chart(pd.DataFrame({'Probability': prediction_proba}, index=['Negative', 'Positive']))

def plot_feature_importance():
    st.write("### Feature Importance")
    feature_importances = model.feature_importances_
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    st.bar_chart(df.set_index('Feature'))

# Streamlit App
st.title("Heart Disease Prediction")
st.write("### Provide the following details to predict the risk of heart disease")

# Input Form
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

# Predict Button
if st.button('Predict', key='predict_button'):
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }

    try:
        data = preprocess_input(input_data)
        prediction = model.predict(data)
        result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
        st.success(f"Prediction: {result}")

        # Plot Prediction Probability
        prediction_proba = model.predict_proba(data)[0]
        plot_probability(prediction_proba)

        # Plot Feature Importance
        plot_feature_importance()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
