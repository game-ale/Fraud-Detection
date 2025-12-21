# Fraud Detection for E-commerce and Bank Transactions

## 📌 Project Overview
Adey Innovations Inc. is a leading financial technology company specializing in e-commerce and banking solutions. This project focuses on building a robust, high-accuracy fraud detection system that balances security with user experience. By leveraging machine learning, geolocation analysis, and transaction pattern recognition, we aim to identify fraudulent activities in real-time, preventing financial loss and building trust with our customers.

## 💼 Business Need
Fraud detection is a critical challenge where:
- **False Positives** (flagging legitimate transactions) alienate customers.
- **False Negatives** (missing actual fraud) lead to direct financial loss.
The goal is to develop models that effectively manage this trade-off while handling the unique challenges of highly imbalanced data.

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blue?style=for-the-badge)

## 📊 Data Description
The project utilizes three primary datasets:
1.  **Fraud_Data.csv (E-commerce Transactions):** Includes user signup details, purchase time, device info, and IP addresses. **Target:** `class`.
2.  **IpAddress_to_Country.csv:** Maps IP addresses to their respective countries for geolocation analysis.
3.  **creditcard.csv (Bank Transactions):** Anonymized PCA-transformed features of bank credit card transactions. **Target:** `Class`.

---

## 🛠️ Project Structure
```text
fraud-detection/
├── .github/workflows/       # CI/CD pipelines
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and feature-engineered data
├── notebooks/               # Jupyter notebooks for EDA and modeling
├── src/                     # Source code (modular logic)
├── tests/                   # Unit tests
├── models/                  # Saved model artifacts
├── scripts/                 # Automation and preprocessing scripts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## ✅ Progress: Task 1 - Data Analysis and Preprocessing
The foundation for modeling has been laid with the following accomplishments:

### 1. Data Cleaning & Integration
- **Missing Values & Duplicates:** Verified data integrity; handled minor missing values in geolocation mapping.
- **Geolocation Integration:** Successfully converted IP addresses to integer ranges and merged the e-commerce dataset with country labels.
- **EDA Notebooks:** Created detailed analysis notebooks:
    - [eda-fraud-data.ipynb](file:///c:/week5/Fraud-Detection/notebooks/eda-fraud-data.ipynb)
    - [eda-creditcard.ipynb](file:///c:/week5/Fraud-Detection/notebooks/eda-creditcard.ipynb)

### 2. Feature Engineering
- **Time-Based Features:** Extracted `hour_of_day`, `day_of_week`, and `time_since_signup` (duration between signup and purchase).
- **Transaction Velocity:** Calculated frequency of transactions per `device_id` and `ip_address` to identify automated or high-risk patterns.

### 3. Handling Class Imbalance
- **Strategy:** Employed **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to address extreme imbalance (especially in the bank dataset).
- **Integrity:** Used stratified splitting to ensure evaluation remains realistic on untouched test data.

### 📈 Key Insights from Task 1
- **Fraud Prevalence:** 9.36% in E-commerce data vs. a mere 0.17% in Credit Card transactions.
- **Top Fraud Countries:** High fraud rates observed in Turkmenistan, Namibia, and Sri Lanka.
- **Transaction Patterns:** Most fraud occurs shortly after user signup, highlighting the importance of the `time_since_signup` feature.

### 4. Data Transformation
- **Encoding:** Categorical variables (Source, Browser, Sex, Country) were encoded for model compatibility.
- **Scaling:** Applied `StandardScaler` to numerical features to ensure balanced influence across and ensemble models.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/game-ale/Fraud-Detection>
   cd Fraud-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Preprocessing
Execute the following scripts in order to generate the processed data:
```bash
python scripts/preprocess_fraud_data.py
python scripts/preprocess_creditcard.py
python scripts/handle_imbalance.py
```

---

## 📅 Roadmap
- [x] **Task 1: Preprocessing & EDA** (Completed)
- [ ] **Task 2: Model Building & Training** (Next)
- [ ] **Task 3: Model Explainability (SHAP)**
- [ ] **Final Reporting & Business Recommendations**

---
**Tutors:** Kerod, Mahbubah, Filimon  
**Data Scientist:** [Gemechu Alemu](https://github.com/game-ale)
