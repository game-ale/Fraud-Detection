import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(csv_path):
    try:
        logger.info(f"Loading credit card data from {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found.")
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_credit_card(df):
    try:
        logger.info("Starting preprocessing for Credit Card Data...")
        
        # Explicit Duplicate Handling
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows.")
            df = df.drop_duplicates()
        
        # Explicit Missing Value Handling
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.info(f"Found {missing} missing values. Filling with median.")
            df = df.fillna(df.median())

        # Standardize 'Amount' and 'Time'
        scaler = StandardScaler()
        df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Credit Card Data")
    parser.add_argument("--input_path", default="data/raw/creditcard.csv", help="Path to input data")
    parser.add_argument("--output_path", default="data/processed/creditcard_processed.csv", help="Output path")
    args = parser.parse_args()
    
    try:
        df = load_data(args.input_path)
        processed_df = preprocess_credit_card(df)
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        processed_df.to_csv(args.output_path, index=False)
        logger.info(f"Processed Credit Card Data saved to {args.output_path}")
    except Exception as e:
        logger.critical(f"Preprocessing failed: {e}")
