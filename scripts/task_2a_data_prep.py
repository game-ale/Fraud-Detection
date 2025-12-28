import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import sys

# Add project root and src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess_fraud_data import load_data as load_fraud, preprocess_fraud_data, encode_features, scale_features
from scripts.preprocess_creditcard import load_data as load_credit, preprocess_credit_card
from src.model_utils import separate_features_target

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_fraud_data(fraud_path, ip_path, output_dir):
    logger.info("Preparing Fraud Data...")
    fraud_df, ip_df = load_fraud(fraud_path, ip_path)
    df = preprocess_fraud_data(fraud_df, ip_df)
    df = encode_features(df)
    df = scale_features(df)
    
    # Drop columns not needed for modeling
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_address_int', 'lower_bound_ip_address', 'upper_bound_ip_address']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    X, y = separate_features_target(df, 'class')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    save_splits(X_train, X_test, y_train, y_test, "fraud", output_dir)

def prepare_creditcard_data(csv_path, output_dir):
    logger.info("Preparing Credit Card Data...")
    df = load_credit(csv_path)
    df = preprocess_credit_card(df)
    
    X, y = separate_features_target(df, 'Class')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    save_splits(X_train, X_test, y_train, y_test, "creditcard", output_dir)

def save_splits(X_train, X_test, y_train, y_test, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, f"{prefix}_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"{prefix}_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"{prefix}_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, f"{prefix}_y_test.csv"), index=False)
    logger.info(f"Saved {prefix} splits to {output_dir}")

if __name__ == "__main__":
    output_dir = "data/processed"
    
    # Fraud Data paths
    fraud_path = "data/raw/Fraud_Data.csv"
    ip_path = "data/raw/IpAddress_to_Country.csv"
    
    # Credit Card Data paths
    credit_path = "data/raw/creditcard.csv"
    
    try:
        if os.path.exists(fraud_path) and os.path.exists(ip_path):
            prepare_fraud_data(fraud_path, ip_path, output_dir)
        else:
            logger.warning("Fraud Data files missing. Skipping.")
            
        if os.path.exists(credit_path):
            prepare_creditcard_data(credit_path, output_dir)
        else:
            logger.warning("Credit Card Data file missing. Skipping.")
            
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)
