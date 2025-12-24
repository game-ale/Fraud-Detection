import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix, 
    classification_report, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import os
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate key metrics for fraud detection."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1,
        'pr_auc': pr_auc
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def perform_cv(model, X, y, cv=5):
    """Perform Stratified K-Fold cross-validation and return mean/std of metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['f1', 'precision', 'recall', 'average_precision']
    
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    summary = {
        'f1_mean': np.mean(cv_results['test_f1']),
        'f1_std': np.std(cv_results['test_f1']),
        'precision_mean': np.mean(cv_results['test_precision']),
        'recall_mean': np.mean(cv_results['test_recall']),
        'pr_auc_mean': np.mean(cv_results['test_average_precision']) # average_precision is proxy for PR AUC
    }
    return summary

def save_model(model, path):
    """Save model artifact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
