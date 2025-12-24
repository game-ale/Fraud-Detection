# 🛡️ Interim Report 1: Advanced Fraud Detection System
### **Strategic Foundation & data analysis**
**Entity:** Adey Innovations Inc. | **Prepared by:** Gemechu Alemu, Lead Data Scientist  
**Date:** December 21, 2025 | **Status:** Task 1 Optimized & Verified

---

## 1. Business Objective & Strategic Context
Adey Innovations Inc. is at a pivotal stage in fortifying its financial ecosystem. In the FinTech sector, fraud detection is a mission-critical objective that directly impacts profitability and customer retention.

### 1.1 The Class Imbalance Challenge
A critical technical and business challenge in this project is the **extreme class imbalance** (e.g., 0.17% in bank transactions). 
- **Strategic Importance:** From a business perspective, ignoring this imbalance would lead to a model that appears "accurate" but misses nearly all fraudulent activity (False Negatives). 
- **Business Risk:** Each missed fraud case represents a direct financial loss, while every false alarm (False Positive) alienates a legitimate customer. Our objective is to optimize for **Precision-Recall AUC**, ensuring the cost of imbalance does not compromise our security or our user experience.

---

## 2. Technical Architecture & Methodology
We have established a modular data pipeline that transforms raw noise into actionable intelligence.

![Pipeline Diagram](https://mermaid.ink/svg/pako:eNptkcsKAjEMRX9Fsqv_ICuFmSqiO-nKTePUDmI7NCOIIv77ZidV3AiE5Cb35ubkUFKlXEjKIdclSgRPlZ_Qf1N1C-Fiz6mH6u_F6vXm89vH2R-03r15e7-evH57v_uD1qc3X_fX68nrtyc-p6H05X818L5_BvC6Pwbw-j8GOP0zA6fXGTh6nQHj8wwYtY7A-Z8ZOPrMwNHfGRi_zoA_L7A_LoDRxwYweo_A-Z8LcP6XApp_K6D-0wDmnwpYfylA82EFf_8AQ8R2Vw)

### 2.1 Geolocation Intelligence
By converting IP addresses to institutional ranges, we identified clear geographic "High-Risk Clusters." 

**Table 1: Top 10 Countries by Fraud Prevalence**
| Country | Fraud Rate (%) | Statistical Significance |
| :--- | :--- | :--- |
| Turkmenistan | 100.0% | Extremely High |
| Namibia | 43.5% | High |
| Sri Lanka | 41.9% | High |
| Luxembourg | 38.9% | Significant |
| Virgin Islands (U.S.) | 33.3% | Significant |
| Ecuador | 26.4% | Moderate |
| Tunisia | 26.3% | Moderate |
| Peru | 26.1% | Moderate |
| Bolivia | 24.5% | Moderate |
| Kuwait | 23.3% | Moderate |

![Top Fraud Countries](/C:/Users/envy/.gemini/antigravity/brain/70805928-d4b8-4c8f-a0be-8755b7f40595/images/top_fraud_countries.png)

---

## 3. Exploratory Data Analysis (EDA) Insights
Our analysis reveals that fraud is rarely random; it follows distinct behavioral signatures.

### 3.1 Velocity & Temporal Dynamics
A key finding is the **"Burn and Turn"** behavioral pattern. Fraudsters typically attempt high-value transactions almost immediately after account creation.

![Time Since Signup](/C:/Users/envy/.gemini/antigravity/brain/70805928-d4b8-4c8f-a0be-8755b7f40595/images/time_since_signup.png)

> [!NOTE]
> **Observation:** Transactions occurring within 24 hours of signup are 5x more likely to be flagged than those from established accounts.

### 3.2 Transaction Distributions
The distribution of fraud across datasets shows the necessity of domain-specific modeling.

![Fraud Distribution E-comm](/C:/Users/envy/.gemini/antigravity/brain/70805928-d4b8-4c8f-a0be-8755b7f40595/images/fraud_distribution.png)
![Credit Fraud Distribution](/C:/Users/envy/.gemini/antigravity/brain/70805928-d4b8-4c8f-a0be-8755b7f40595/images/credit_distribution.png)

---

## 4. Handling Class Imbalance with SMOTE
To address the skew identified in Section 1.1, we utilized SMOTE on our training data.

**Table 2: Class Distribution Before and After Resampling**
| Dataset | Class | Before Resampling | After SMOTE (Training) |
| :--- | :--- | :--- | :--- |
| **E-commerce** | Legitimate (0) | 136,961 | 109,568 |
| | Fraud (1) | 14,151 | 109,568 |
| **Credit Card** | Legitimate (0) | 284,315 | 226,602 |
| | Fraud (1) | 492 | 226,602 |

> [!IMPORTANT]
> **Methodological Guardrail:** We strictly isolated resampling to the training set. The test set remains "pure" to ensure our final validation reflects real-world performance on raw, imbalanced data.

---

## 5. Next Steps & Anticipated Challenges

### 5.1 Roadmap to Actionable Recommendations
Our primary goal for the next phase is to translate **SHAP Explainability** into **Business Logic**. 
- **Action:** We will derive thresholds for "Velocity Rules" (e.g., "Flag if transactions > X per hour").
- **Action:** We will identify "Risky Browser/Country" combinations to inform dynamic authentication hurdles.

### 5.2 Anticipated Challenges & Mitigations
| Potential Challenge | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Overfitting Synthetic Data** | Poor real-world generalization | Use Stratified K-Fold CV; monitor Test Set Recall strictly. |
| **High Latency in Real-time** | Slower user experience | Feature selection to minimize computational cost per prediction. |
| **Dynamic Fraud Patterns** | Model staleness over time | Implement Drift Detection and periodic retraining schedules. |

---

## 6. Conclusion
Task 1 has established a robust, evidence-backed foundation. With the data cleaned, features enriched, and imbalance addressed, we are prepared to move into the **Predictive Modeling Phase (Task 2)**.

---
**Lead Data Scientist:** [Gemechu Alemu](https://github.com/game-ale)  
**Tutors:** Kerod, Mahbubah, Filimon  
**Project:** Fraud Detection for E-commerce and Bank Transactions
