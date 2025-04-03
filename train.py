import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model import ChurnPredictor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    # Load your dataset
    try:
        df = pd.read_csv('customer_data.csv')
    except FileNotFoundError:
        print("Please provide a customer_data.csv file with the following columns:")
        print("- customerID")
        print("- gender")
        print("- SeniorCitizen")
        print("- Partner")
        print("- Dependents")
        print("- tenure")
        print("- PhoneService")
        print("- MultipleLines")
        print("- InternetService")
        print("- OnlineSecurity")
        print("- OnlineBackup")
        print("- DeviceProtection")
        print("- TechSupport")
        print("- StreamingTV")
        print("- StreamingMovies")
        print("- Contract")
        print("- PaperlessBilling")
        print("- PaymentMethod")
        print("- MonthlyCharges")
        print("- TotalCharges")
        print("- Churn (target variable)")
        return
    
    # Separate features and target
    X = df.drop(['Churn'], axis=1)  # customerID will be dropped in preprocessing
    y = df['Churn']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Save the label encoder for later use
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess the data
    X_processed = preprocessor.preprocess_data(X)
    
    # Split the data
    X_train, X_temp, y_train, y_temp = preprocessor.split_data(X_processed, y)
    X_val, X_test, y_val, y_test = preprocessor.split_data(X_temp, y_temp)
    
    # Initialize and train the model
    print("Training model...")
    model = ChurnPredictor()
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Save the model and preprocessor
    print("\nSaving model and preprocessor...")
    model.save_model('churn_model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Model and preprocessor saved successfully!")

if __name__ == "__main__":
    main() 