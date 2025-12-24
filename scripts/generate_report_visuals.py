import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

def generate_visuals():
    # Load data
    fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')

    # Preprocess (similar to scripts/preprocess_fraud_data.py)
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600 # hours
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    fraud_df['device_freq'] = fraud_df.groupby('device_id')['user_id'].transform('count')
    fraud_df['ip_freq'] = fraud_df.groupby('ip_address')['user_id'].transform('count')

    # Create images directory
    img_dir = 'images/report'
    os.makedirs(img_dir, exist_ok=True)

    # 1. Time Since Signup Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=fraud_df, x='time_since_signup', hue='class', bins=50, kde=True, element="step")
    plt.title('Distribution of Time Since Signup (Hours)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Frequency')
    plt.savefig(f'{img_dir}/time_since_signup_dist.png')
    plt.close()

    # Zoom in on first 24 hours
    plt.figure(figsize=(10, 6))
    sns.histplot(data=fraud_df[fraud_df['time_since_signup'] < 24], x='time_since_signup', hue='class', bins=24, kde=True)
    plt.title('Time Since Signup - First 24 Hours')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    plt.savefig(f'{img_dir}/time_since_signup_zoom.png')
    plt.close()

    # 2. Fraud Rate by Hour of Day
    plt.figure(figsize=(10, 6))
    hour_fraud = fraud_df.groupby('hour_of_day')['class'].mean().reset_index()
    sns.barplot(x='hour_of_day', y='class', data=hour_fraud, palette='magma')
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate')
    plt.savefig(f'{img_dir}/fraud_rate_hour.png')
    plt.close()

    # 3. Fraud Rate by Day of Week
    plt.figure(figsize=(10, 6))
    day_fraud = fraud_df.groupby('day_of_week')['class'].mean().reset_index()
    sns.barplot(x='day_of_week', y='class', data=day_fraud, palette='viridis')
    plt.title('Fraud Rate by Day of Week (0=Monday)')
    plt.xlabel('Day of Week')
    plt.ylabel('Fraud Rate')
    plt.savefig(f'{img_dir}/fraud_rate_day.png')
    plt.close()

    # 4. Device Frequency Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='device_freq', hue='class', data=fraud_df[fraud_df['device_freq'] > 1])
    plt.title('Fraud by Device Frequency (>1)')
    plt.xlabel('Transactions per Device')
    plt.ylabel('Count')
    plt.savefig(f'{img_dir}/device_freq_fraud.png')
    plt.close()

    # 5. Univariate: Purchase Value
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_df['purchase_value'], bins=50, kde=True, color='blue')
    plt.title('Purchase Value Distribution (Univariate)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'{img_dir}/purchase_value_dist.png')
    plt.close()

    print(f"Visualizations saved to {img_dir}")

if __name__ == "__main__":
    generate_visuals()
