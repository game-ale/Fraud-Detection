import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_imbalance(csv_path, target_col, output_prefix):
    try:
        logger.info(f"Handling imbalance for {csv_path}...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found.")
            
        df = pd.read_csv(csv_path)
        
        # Drop non-numeric for SMOTE if any left (e.g. times, IDs)
        numeric_df = df.select_dtypes(include=[np.number])
        # SMOTE doesn't handle NaNs
        if numeric_df.isnull().any().any():
            logger.warning("Found NaNs in numeric data. Filling with 0.")
            numeric_df = numeric_df.fillna(0)
            
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"Before SMOTE - {output_prefix} training set distribution: {y_train.value_counts().to_dict()}")
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - {output_prefix} training set distribution: {y_train_res.value_counts().to_dict()}")
        
        # Save split data
        os.makedirs('data/processed', exist_ok=True)
        X_train_res.to_csv(f'data/processed/{output_prefix}_X_train.csv', index=False)
        X_test.to_csv(f'data/processed/{output_prefix}_X_test.csv', index=False)
        y_train_res.to_csv(f'data/processed/{output_prefix}_y_train.csv', index=False)
        y_test.to_csv(f'data/processed/{output_prefix}_y_test.csv', index=False)
        
        logger.info(f"Saved {output_prefix} split files to data/processed/")
    except Exception as e:
        logger.error(f"Error handling imbalance for {output_prefix}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle Class Imbalance")
    parser.add_argument("--fraud_csv", default="data/processed/fraud_data_processed.csv", help="Path to processed fraud data")
    parser.add_argument("--credit_csv", default="data/processed/creditcard_processed.csv", help="Path to processed credit card data")
    args = parser.parse_args()
    
    try:
        # Fraud Data
        handle_imbalance(args.fraud_csv, 'class', 'fraud')
        
        # Credit Card Data
        handle_imbalance(args.credit_csv, 'Class', 'creditcard')
    except Exception as e:
        logger.critical(f"Imbalance handling failed: {e}")
