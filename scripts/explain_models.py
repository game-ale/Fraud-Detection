import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import argparse
from sklearn.metrics import confusion_matrix
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data_and_model(dataset_name):
    """Load model and test data for a given dataset."""
    model_path = f'models/{dataset_name}_best_model.joblib'
    X_test_path = f'data/processed/{dataset_name}_X_test.csv'
    y_test_path = f'data/processed/{dataset_name}_y_test.csv'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    return model, X_test, y_test

def plot_builtin_importance(model, feature_names, dataset_name, save_dir):
    """Extract and plot built-in feature importance."""
    # Handle different model types (XGBoost vs Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For Logistic Regression, use coefficients
        importances = np.abs(model.coef_[0])
        
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top 10 Built-in Feature Importances ({dataset_name})')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_builtin_importance.png'))
    plt.close()

def generate_shap_plots(model, X_test, y_test, dataset_name, save_dir):
    """Generate SHAP summary and individual force plots."""
    # SHAP can be slow on large datasets, so we sample
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # Initialize Explainer
    # TreeExplainer is best for XGB/RF
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot handles multi-class/binary differently 
    # For binary, it often returns a list [neg_values, pos_values]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # Print top 10 SHAP features to console for reporting
    try:
        feature_importance = np.abs(shap_values_to_plot).mean(0)
        # Ensure it's 1D
        if len(feature_importance.shape) > 1:
             feature_importance = feature_importance.flatten()
             
        top_features = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"\nTop 10 SHAP features for {dataset_name}:")
        print(top_features)
    except Exception as e:
        print(f"Could not print top features: {e}")
        print(f"Shape of shap_values_to_plot: {np.shape(shap_values_to_plot)}")

    # 1. SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    # In newer SHAP versions, summary_plot might prefer a specific call pattern
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    plt.title(f'SHAP Summary Plot ({dataset_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_shap_summary.png'))
    plt.close()
    
    # 2. Individual Force Plots (TP, FP, FN)
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Identify indices
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"Dataset {dataset_name}: Found {len(tp_idx)} TP, {len(fp_idx)} FP, {len(fn_idx)} FN cases.")
    
    cases = {
        'true_positive': tp_idx[0] if len(tp_idx) > 0 else None,
        'false_positive': fp_idx[0] if len(fp_idx) > 0 else None,
        'false_negative': fn_idx[0] if len(fn_idx) > 0 else None
    }
    
    # Get expected value (base value)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1] # Use positive class for binary

    for case_name, idx in cases.items():
        if idx is not None:
            # Re-calculate SHAP for the specific instance
            instance = X_test.iloc[[idx]]
            sv = explainer.shap_values(instance)
            
            # Select correct class for binary
            if isinstance(sv, list) and len(sv) == 2:
                sv_to_plot = sv[1]
            else:
                sv_to_plot = sv
                
            plt.figure(figsize=(20, 3))
            # matplotlib=True is for individual plots
            try:
                shap.force_plot(
                    base_val, 
                    sv_to_plot[0] if len(sv_to_plot.shape) > 1 else sv_to_plot, 
                    instance.iloc[0], 
                    matplotlib=True, 
                    show=False
                )
                plt.title(f'SHAP Force Plot: {case_name} ({dataset_name})', y=1.5)
                save_path = os.path.join(save_dir, f'{dataset_name}_shap_{case_name}.png')
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved {case_name} plot to {save_path}")
            except Exception as e:
                print(f"Error plotting {case_name}: {e}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Explainability with SHAP")
    parser.add_argument("--dataset", choices=['fraud', 'creditcard'], required=True)
    args = parser.parse_args()
    
    save_dir = 'models/explainability'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting explainability analysis for {args.dataset}...")
    model, X_test, y_test = load_data_and_model(args.dataset)
    
    print("Generating built-in importance plot...")
    plot_builtin_importance(model, X_test.columns, args.dataset, save_dir)
    
    print("Generating SHAP plots (this may take a minute)...")
    generate_shap_plots(model, X_test, y_test, args.dataset, save_dir)
    
    print(f"Analysis complete. Plots saved to {save_dir}")
