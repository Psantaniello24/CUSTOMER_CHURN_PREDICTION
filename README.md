# Customer Churn Prediction System

This project implements a customer churn prediction system using XGBoost and Streamlit. The system includes data preprocessing, model training with hyperparameter optimization, and a web interface for real-time predictions.

## Video example

![Demo GIF](./demo_retail.gif)


## Try it at : 


## Features

- Data preprocessing with handling of missing values and categorical encoding
- XGBoost model with hyperparameter optimization using Optuna
- Model evaluation using precision, recall, and AUC-ROC metrics
- Interactive Streamlit web interface for real-time predictions
- Feature importance visualization

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train.py
```

2. Run the Streamlit app locally:
```bash
streamlit run app.py
```

## Project Structure

- `data_preprocessing.py`: Contains the data preprocessing pipeline
- `model.py`: Implements the XGBoost model with hyperparameter optimization
- `train.py`: Script to train and save the model
- `app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies

## Input Features

The model requires the following customer information:
- Customer ID
- Gender
- Senior Citizen status
- Partner status
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

## Model Performance

The model is evaluated using:
- Precision
- Recall
- AUC-ROC
