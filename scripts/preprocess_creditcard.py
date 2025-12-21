import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv('data/raw/creditcard.csv')
    return df

def preprocess_credit_card(df):
    print("Preprocessing Credit Card Data...")
    
    # Dataset is already pre-clean and mostly PCA features
    # Standardize 'Amount' and 'Time'
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    
    # Class imbalance is extreme (0.17%), so we don't resample here.
    # Resampling should only be done on training data.
    
    return df

if __name__ == "__main__":
    df = load_data()
    processed_df = preprocess_credit_card(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv('data/processed/creditcard_processed.csv', index=False)
    print("Processed Credit Card Data saved to data/processed/creditcard_processed.csv")
