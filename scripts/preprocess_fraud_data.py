import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(fraud_path, ip_path):
    try:
        logger.info(f"Loading data from {fraud_path} and {ip_path}")
        if not os.path.exists(fraud_path) or not os.path.exists(ip_path):
            raise FileNotFoundError("One or more data files not found.")
        
        fraud_df = pd.read_csv(fraud_path)
        ip_df = pd.read_csv(ip_path)
        return fraud_df, ip_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_fraud_data(df, ip_df):
    try:
        logger.info("Starting preprocessing for Fraud Data...")
        
        # Explicit Duplicate Handling
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows.")
            df = df.drop_duplicates()
        
        # Explicit Missing Value Handling
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.info(f"Found {missing} missing values. Dropping rows with missing values.")
            df = df.dropna()

        # 1. IP processing
        df['ip_address_int'] = df['ip_address'].astype('float64')
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('float64')
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('float64')
        
        # 2. Merge with IP to Country mapping
        ip_df = ip_df.sort_values('lower_bound_ip_address')
        df = df.sort_values('ip_address_int')
        
        df = pd.merge_asof(df, ip_df, left_on='ip_address_int', right_on='lower_bound_ip_address')
        
        df.loc[df['ip_address_int'] > df['upper_bound_ip_address'], 'country'] = 'Unknown'
        df['country'] = df['country'].fillna('Unknown')
        
        # 3. Time-based features
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # 4. Transaction frequency/velocity
        df['device_freq'] = df.groupby('device_id')['user_id'].transform('count')
        df['ip_freq'] = df.groupby('ip_address')['user_id'].transform('count')
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def encode_features(df):
    try:
        categorical_cols = ['source', 'browser', 'sex', 'country']
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
    except Exception as e:
        logger.error(f"Error in encoding: {e}")
        raise

def scale_features(df):
    try:
        numerical_cols = ['purchase_value', 'time_since_signup', 'hour_of_day', 'day_of_week', 'device_freq', 'ip_freq', 'age']
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df
    except Exception as e:
        logger.error(f"Error in scaling: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Fraud Data")
    parser.add_argument("--fraud_path", default="data/raw/Fraud_Data.csv", help="Path to fraud data")
    parser.add_argument("--ip_path", default="data/raw/IpAddress_to_Country.csv", help="Path to IP mapping data")
    parser.add_argument("--output_path", default="data/processed/fraud_data_processed.csv", help="Output path")
    args = parser.parse_args()
    
    try:
        fraud_df, ip_df = load_data(args.fraud_path, args.ip_path)
        processed_df = preprocess_fraud_data(fraud_df, ip_df)
        processed_df = encode_features(processed_df)
        processed_df = scale_features(processed_df)
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        processed_df.to_csv(args.output_path, index=False)
        logger.info(f"Processed Fraud Data saved to {args.output_path}")
    except Exception as e:
        logger.critical(f"Preprocessing failed: {e}")
