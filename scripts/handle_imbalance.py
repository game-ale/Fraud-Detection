import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def handle_imbalance(csv_path, target_col, output_prefix):
    print(f"Handling imbalance for {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop non-numeric for SMOTE if any left (e.g. times, IDs)
    numeric_df = df.select_dtypes(include=[np.number])
    # SMOTE doesn't handle NaNs, and we have some in IP bounds for 'Unknown' countries
    numeric_df = numeric_df.fillna(0)
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Before SMOTE - {output_prefix} training set distribution:")
    print(y_train.value_counts())
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - {output_prefix} training set distribution:")
    print(y_train_res.value_counts())
    
    # Save split data
    os.makedirs('data/processed', exist_ok=True)
    X_train_res.to_csv(f'data/processed/{output_prefix}_X_train.csv', index=False)
    X_test.to_csv(f'data/processed/{output_prefix}_X_test.csv', index=False)
    y_train_res.to_csv(f'data/processed/{output_prefix}_y_train.csv', index=False)
    y_test.to_csv(f'data/processed/{output_prefix}_y_test.csv', index=False)
    
    print(f"Saved {output_prefix} split files to data/processed/")

if __name__ == "__main__":
    # Fraud Data
    handle_imbalance('data/processed/fraud_data_processed.csv', 'class', 'fraud')
    
    # Credit Card Data
    handle_imbalance('data/processed/creditcard_processed.csv', 'Class', 'creditcard')
