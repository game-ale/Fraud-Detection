# 🛡️ Fraud Detection for E-commerce and Bank Transactions

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)](https://xgboost.ai/)
[![DeepMind](https://img.shields.io/badge/Built%20by-Adey%20Innovations%20Inc.-blue?style=for-the-badge)](https://github.com/game-ale)

## 📌 Project Overview
Adey Innovations Inc. specializes in secure financial technology. This project develops a high-accuracy fraud detection system for both e-commerce and banking sectors. By blending **Geolocation Analysis**, **Transaction Pattern Recognition**, and **Advanced Ensembling**, we minimize financial leakage while maintaining a seamless user experience.

## 💼 Business Need
Fraud detection is a balancing act:
- **Minimize False Positives:** Avoid blocking legitimate customers.
- **Minimize False Negatives:** Prevent direct financial loss from missed fraud.
Our models are optimized for **Recall** and **PR-AUC** to maximize security without compromising trust.

---

## 🏗️ Project Structure
| Folder | Description |
| :--- | :--- |
| `data/` | Raw and engineered datasets (Gitignored). |
| `notebooks/` | EDA, Feature Engineering, and Modeling experiments. |
| `src/` | Modular source code and reusable utilities. |
| `models/` | Saved model artifacts (.joblib) and confusion matrices. |
| `reports/` | Official project reports and documentation. |
| `scripts/` | Automation for preprocessing, splitting, and training. |
| `tests/` | Unit tests for core logic verification. |

---

## ✅ Progress: Phase 1 & 2

### 📋 Task 1: Data Analysis and Preprocessing
- **Geolocation Mapping:** Merged IP address ranges with country data for spatial fraud insights.
- **Feature Engineering:** Extracted `time_since_signup`, `hour_of_day`, and transaction frequency features.
- **Imbalance Handling:** Applied **SMOTE** to handle extreme class imbalance (0.17% fraud in banking).

### 🤖 Task 2: Model Building and Training
We evaluated multiple architectures using **Stratified 5-Fold Cross-Validation**.

#### **Best Model Performance (XGBoost)**
| Dataset | Precision | Recall | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **E-commerce (Fraud)** | 0.95 | 0.55 | 0.70 | 0.78 |
| **Bank (Credit Card)** | 0.53 | 0.98 | 0.69 | 0.81 |

> [!IMPORTANT]
> For banking transactions, we achieved a **98% Recall**, ensuring nearly all fraudulent activities are captured, which is critical for financial security.

#### **Model Comparison**
- **Logistic Regression:** Served as a strong, interpretable baseline.
- **Random Forest:** Improved performance through ensemble bagging.
- **XGBoost:** Selected as final model due to its superior gradient boosting capability and PR-AUC performance.

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/game-ale/Fraud-Detection.git
cd Fraud-Detection
pip install -r requirements.txt
```

### Execution Pipeline
1. **Preprocessing:** `python scripts/preprocess_fraud_data.py`
2. **Modeling:** `python scripts/train_models.py --dataset fraud --dataset creditcard`

---

## 🧠 Phase 3: Model Explainability (SHAP)
We moved beyond "black-box" predictions to understand the *why* behind fraud detection.

### Key Drivers
- **Time Since Signup:** Immediate transactions after account creation are the #1 predictor of fraud.
- **Purchase Value:** High-value transactions, especially from new IPs, carry significant weight.
- **Anonymized Features (V14, V17):** In the bank dataset, specific PCA vectors (likely correlating to card-not-present scenarios) are dominant.

> [!TIP]
> **Business Rule:** We recommend implementing **Step-Up Authentication** (OTP) for any transaction > $100 occurring within 1 hour of account signup.

[👉 Read the Full Explainability Report](reports/model_explainability_report.md)

---

## 📅 Roadmap
- [x] **Task 1: Preprocessing & EDA**
- [x] **Task 2: Model Building & Training**
- [x] **Task 3: Model Explainability (SHAP)**
- [ ] **Task 4: Model Deployment & API**
- [ ] **Task 5: Final Dashboard & Reporting**

---
**Tutors:** Kerod, Mahbubah, Filimon  
**Data Scientist:** [Gemechu Alemu](https://github.com/game-ale)
