import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='mean')
        
    def preprocess_data(self, df):
        """Preprocess the input dataframe"""
        # Create a copy to avoid modifying the original data
        df_processed = df.copy()
        
        # Drop customerID if present as it's not a feature
        if 'customerID' in df_processed.columns:
            df_processed = df_processed.drop('customerID', axis=1)
        
        # Handle TotalCharges - convert to numeric and handle spaces
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'].replace(' ', np.nan))
        
        # Identify numeric and categorical columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_columns = [col for col in df_processed.columns if col not in numeric_columns]
        
        # Handle missing values in numeric columns
        df_processed[numeric_columns] = self.numeric_imputer.fit_transform(df_processed[numeric_columns])
        
        # Handle categorical columns
        for col in categorical_columns:
            # Fill missing values with mode
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            # Encode categorical variables
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Scale numeric features
        df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
        
        return df_processed
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def transform_new_data(self, df):
        """Transform new data using the fitted preprocessor"""
        df_processed = df.copy()
        
        # Drop customerID if present as it's not a feature
        if 'customerID' in df_processed.columns:
            df_processed = df_processed.drop('customerID', axis=1)
        
        # Handle TotalCharges - convert to numeric and handle spaces
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'].replace(' ', np.nan))
        
        # Identify numeric and categorical columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_columns = [col for col in df_processed.columns if col not in numeric_columns]
        
        # Handle missing values in numeric columns
        df_processed[numeric_columns] = self.numeric_imputer.transform(df_processed[numeric_columns])
        
        # Handle categorical columns
        for col in categorical_columns:
            # Fill missing values with mode
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            # Encode categorical variables using fitted encoders
            if col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Scale numeric features
        df_processed[numeric_columns] = self.scaler.transform(df_processed[numeric_columns])
        
        return df_processed 