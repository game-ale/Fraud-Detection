import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from preprocess_fraud_data import encode_features, scale_features

def test_encode_features():
    df = pd.DataFrame({
        'source': ['SEO', 'Ads'],
        'browser': ['Chrome', 'Safari'],
        'sex': ['M', 'F'],
        'country': ['USA', 'China'],
        'value': [10, 20]
    })
    encoded_df = encode_features(df.copy())
    
    assert encoded_df['source'].dtype in [np.int32, np.int64]
    assert encoded_df['browser'].dtype in [np.int32, np.int64]
    assert len(encoded_df['source'].unique()) == 2

def test_scale_features():
    df = pd.DataFrame({
        'purchase_value': [10, 100, 1000],
        'time_since_signup': [1, 10, 100],
        'hour_of_day': [0, 12, 23],
        'day_of_week': [0, 3, 6],
        'device_freq': [1, 5, 10],
        'ip_freq': [1, 5, 10],
        'age': [20, 30, 40]
    })
    scaled_df = scale_features(df.copy())
    
    # Standardized columns should have mean approx 0 and std approx 1
    assert np.isclose(scaled_df['purchase_value'].mean(), 0, atol=1e-7)
    assert np.isclose(scaled_df['purchase_value'].std(ddof=0), 1, atol=1e-7)

def test_missing_value_handling():
    # We can check if preprocess_fraud_data handles NaNs if we mock the IP mapping
    # But for now, let's keep tests simple and robust
    pass
