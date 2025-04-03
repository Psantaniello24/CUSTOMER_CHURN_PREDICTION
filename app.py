import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model import ChurnPredictor
import joblib

def main():
    st.title("Customer Churn Prediction")
    st.write("This application predicts customer churn using XGBoost model.")
    
    # Load the trained model and preprocessor
    try:
        model = ChurnPredictor()
        model.load_model('churn_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
    except:
        st.error("Please train the model first using the training script.")
        return
    
    # Create input fields for features
    st.subheader("Enter Customer Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        customerID = st.text_input("Customer ID")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0)
        
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Create a button for prediction
    if st.button("Predict Churn"):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'customerID': [customerID],
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Preprocess the input data
        processed_data = preprocessor.transform_new_data(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Convert prediction back to original label
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction_label == "Yes":
            st.error(f"High Churn Risk (Probability: {probability:.2%})")
        else:
            st.success(f"Low Churn Risk (Probability: {probability:.2%})")
        
        # Display feature importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'feature': processed_data.columns,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.bar_chart(importance.set_index('feature'))

if __name__ == "__main__":
    main() 