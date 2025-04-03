import xgboost as xgb
import optuna
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import joblib
import numpy as np

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization"""
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        }
        
        model = xgb.XGBClassifier(**param, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Return the average of metrics as the objective value
        return (precision + recall + auc) / 3
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Train the model with hyperparameter optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                      n_trials=n_trials)
        
        self.best_params = study.best_params
        self.model = xgb.XGBClassifier(**self.best_params, random_state=42)
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions on new data"""
        return self.model.predict_proba(X)
    
    def save_model(self, model_path):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        saved_model = joblib.load(model_path)
        self.model = saved_model['model']
        self.best_params = saved_model['best_params'] 