import pandas as pd
import numpy as np
import os
import argparse
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_utils import calculate_metrics, plot_confusion_matrix, perform_cv, save_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_split_data(prefix):
    """Load pre-processed and split data."""
    X_train = pd.read_csv(f'data/processed/{prefix}_X_train.csv')
    X_test = pd.read_csv(f'data/processed/{prefix}_X_test.csv')
    y_train = pd.read_csv(f'data/processed/{prefix}_y_train.csv').values.ravel()
    y_test = pd.read_csv(f'data/processed/{prefix}_y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_evaluate(dataset_name, X_train, X_test, y_train, y_test):
    results = []
    
    # 1. Baseline: Logistic Regression
    logger.info(f"Training Logistic Regression for {dataset_name}...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_cv = perform_cv(lr, X_train, y_train)
    lr.fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_test)
    y_probs_lr = lr.predict_proba(X_test)[:, 1]
    lr_metrics = calculate_metrics(y_test, y_pred_lr, y_probs_lr)
    
    results.append({
        'model': 'Logistic Regression',
        'cv_f1_mean': lr_cv['f1_mean'],
        'test_f1': lr_metrics['f1_score'],
        'test_pr_auc': lr_metrics['pr_auc']
    })
    
    # 2. Ensemble: Random Forest
    logger.info(f"Training Random Forest for {dataset_name}...")
    rf_params = {
        'n_estimators': [50 if dataset_name == 'creditcard' else 100],
        'max_depth': [10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, rf_params, cv=2, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    
    rf_cv = perform_cv(best_rf, X_train, y_train, cv=2 if dataset_name == 'creditcard' else 3)
    y_pred_rf = best_rf.predict(X_test)
    y_probs_rf = best_rf.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_metrics(y_test, y_pred_rf, y_probs_rf)
    
    results.append({
        'model': 'Random Forest',
        'cv_f1_mean': rf_cv['f1_mean'],
        'test_f1': rf_metrics['f1_score'],
        'test_pr_auc': rf_metrics['pr_auc']
    })
    
    # 3. Ensemble: XGBoost
    logger.info(f"Training XGBoost for {dataset_name}...")
    xgb_params = {
        'n_estimators': [50 if dataset_name == 'creditcard' else 100],
        'max_depth': [3],
        'learning_rate': [0.1]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid_xgb = GridSearchCV(xgb, xgb_params, cv=2, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    
    xgb_cv = perform_cv(best_xgb, X_train, y_train, cv=2 if dataset_name == 'creditcard' else 3)
    y_pred_xgb = best_xgb.predict(X_test)
    y_probs_xgb = best_xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = calculate_metrics(y_test, y_pred_xgb, y_probs_xgb)
    
    results.append({
        'model': 'XGBoost',
        'cv_f1_mean': xgb_cv['f1_mean'],
        'test_f1': xgb_metrics['f1_score'],
        'test_pr_auc': xgb_metrics['pr_auc']
    })
    
    # Save best model based on test F1 and plot confusion matrix
    best_model_info = max(results, key=lambda x: x['test_f1'])
    logger.info(f"Best model for {dataset_name}: {best_model_info['model']} with F1: {best_model_info['test_f1']}")
    
    if best_model_info['model'] == 'Logistic Regression':
        best_model = lr
        best_preds = y_pred_lr
    elif best_model_info['model'] == 'Random Forest':
        best_model = best_rf
        best_preds = y_pred_rf
    else:
        best_model = best_xgb
        best_preds = y_pred_xgb
    
    save_model(best_model, f'models/{dataset_name}_best_model.joblib')
    plot_confusion_matrix(y_test, best_preds, f"{best_model_info['model']} on {dataset_name}", save_path=f"models/{dataset_name}_confusion_matrix.png")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Models")
    parser.add_argument("--dataset", choices=['fraud', 'creditcard'], required=True, help="Dataset to train on")
    args = parser.parse_args()
    
    try:
        X_train, X_test, y_train, y_test = load_split_data(args.dataset)
        results = train_and_evaluate(args.dataset, X_train, X_test, y_train, y_test)
        
        # Print results table
        print(f"\nResults for {args.dataset} dataset:")
        print(pd.DataFrame(results))
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
