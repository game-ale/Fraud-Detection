import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data():
    fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    return fraud_df, ip_df

def ip_to_int(ip):
    try:
        parts = list(map(int, ip.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return 0

def preprocess_fraud_data(df, ip_df):
    print("Preprocessing Fraud Data...")
    
    # 1. Convert IP to integer and ensure dtype match for merge_asof
    df['ip_address_int'] = df['ip_address'].apply(ip_to_int).astype('float64')
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('float64')
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('float64')
    
    # 2. Merge with IP to Country mapping
    # Sort ip_df for merge_asof
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    df = df.sort_values('ip_address_int')
    
    # merge_asof defaults to 'backward', matching lower_bound_ip_address
    df = pd.merge_asof(df, ip_df, left_on='ip_address_int', right_on='lower_bound_ip_address')
    
    # Validate result: only keep if within range
    df.loc[df['ip_address_int'] > df['upper_bound_ip_address'], 'country'] = 'Unknown'
    df['country'] = df['country'].fillna('Unknown')
    
    # 3. Time-based features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # 4. Transaction frequency/velocity
    # Number of transactions per device
    df['device_freq'] = df.groupby('device_id')['user_id'].transform('count')
    # Number of transactions per IP
    df['ip_freq'] = df.groupby('ip_address')['user_id'].transform('count')
    
    # 5. Clean up columns
    # We'll drop original times and non-numeric for now, or encode them
    return df

def encode_features(df):
    categorical_cols = ['source', 'browser', 'sex', 'country']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(df):
    numerical_cols = ['purchase_value', 'time_since_signup', 'hour_of_day', 'day_of_week', 'device_freq', 'ip_freq', 'age']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    fraud_df, ip_df = load_data()
    processed_df = preprocess_fraud_data(fraud_df, ip_df)
    processed_df = encode_features(processed_df)
    processed_df = scale_features(processed_df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv('data/processed/fraud_data_processed.csv', index=False)
    print("Processed Fraud Data saved to data/processed/fraud_data_processed.csv")
