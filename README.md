# Fraud Detection using Isolation Forest (PySpark)

A fraud detection pipeline built with **PySpark** and **SynapseML Isolation Forest** using a realistic synthetic banking transaction dataset. This project demonstrates how unsupervised machine learning can identify anomalous financial transactions through behavioral feature engineering without requiring labeled fraud data.

---

# Project Overview

Financial fraud is highly imbalanced, making supervised learning difficult due to the scarcity of fraud labels. This project addresses the problem by implementing an **unsupervised anomaly detection pipeline** based on the Isolation Forest algorithm.

Instead of relying on transaction labels, the model learns normal customer behavior from historical transaction patterns and isolates observations that significantly deviate from those patterns.

The project emphasizes **behavioral feature engineering**, where each transaction is transformed into customer-centric behavioral indicators before anomaly detection.

---

# Objectives

* Build an end-to-end fraud detection pipeline using PySpark.
* Simulate realistic banking transaction behavior.
* Engineer behavioral features from raw transaction history.
* Normalize features using StandardScaler.
* Detect anomalies using Isolation Forest.
* Rank suspicious transactions based on anomaly severity.

---

# Project Structure

```
.
├── data/
│   ├── realistic_transaction_dataset_38484.csv
│   └── realistic_transaction_dataset_38484.xlsx
│
├── notebook/
│   └── isolation_forest_module_synthetics data.ipynb
│
├── logging/
│
└── README.md
```

---

# Dataset

The project uses a **synthetic banking transaction dataset** containing **38,484 transactions**.

Unlike randomly generated data, each customer follows a relatively consistent behavioral profile, allowing meaningful behavioral feature engineering.

## Raw Features

| Feature             | Description              |
| ------------------- | ------------------------ |
| TransactionDatetime | Transaction timestamp    |
| AccountCIFNo        | Customer identifier      |
| TransactionAmount   | Transaction amount (IDR) |

---

# Synthetic Customer Behaviour

Each customer belongs to one behavioral persona.

* Dormant Customer
* Regular Customer
* Active Customer
* Business Customer
* Wealth Customer

Each persona has different:

* transaction frequency
* spending habits
* active transaction hours
* spending volatility
* seasonal behaviour

The dataset also simulates:

* Salary-day spending
* Weekend effect
* Year-end spending
* Natural spending drift
* Rare high-value purchases

No fraud labels are included because the project focuses on **unsupervised anomaly detection**.

---

# Feature Engineering

The following behavioral features are generated.

| Feature                      | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| hour_in_dec                  | Continuous representation of transaction time         |
| time_last_transaction        | Time interval since previous transaction              |
| amt_transaction_hourly       | Total transaction amount per customer within one hour |
| amt_cumulative               | Running cumulative transaction amount                 |
| amt_diff_last_transaction    | Difference from previous transaction amount           |
| amt_relative_transaction     | Transaction amount relative to customer average       |
| transaction_freq_daily       | Number of daily transactions                          |
| transaction_freq_variability | Variability of hourly transaction frequency           |

These features describe **customer behaviour** instead of transaction values alone.

---

# Pipeline

```
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Vector Assembler
        │
        ▼
StandardScaler
        │
        ▼
Isolation Forest
        │
        ▼
Anomaly Score
        │
        ▼
Outlier Ranking
```

---

# Technologies

* Python
* PySpark
* SynapseML
* Isolation Forest
* Spark ML Pipeline
* Pandas
* NumPy
* OpenPyXL

---

# Isolation Forest Configuration

The model is implemented using **SynapseML Isolation Forest**.

Pipeline stages include:

1. VectorAssembler
2. StandardScaler
3. IsolationForest

Output columns:

* anomalyScore
* predictedLabel
* scaledFeatures

---

# Results

The notebook produces:

* descriptive statistics
* anomaly score distribution
* anomaly ranking
* normal vs anomaly summary
* top suspicious transactions

The anomaly score can be used as a risk indicator for further investigation.

---

# Why Isolation Forest?

Isolation Forest is particularly suitable because:

* does not require fraud labels
* handles highly imbalanced data
* scales well for large datasets
* computationally efficient in Spark
* suitable for financial anomaly detection

---

# How to Run

## Clone repository

```bash
git clone <repository-url>
cd <repository-name>
```

## Install dependencies

```bash
pip install pandas numpy pyspark openpyxl
```

Install SynapseML according to your Spark version.

## Prepare dataset

Place the synthetic dataset inside the `data/` directory.

## Execute notebook

Run:

```
isolation_forest_module_synthetics data.ipynb
```

The notebook performs:

1. Spark initialization
2. Data ingestion
3. Feature engineering
4. Standardization
5. Isolation Forest training
6. Result summarization

---

# Future Improvements

* Time-series customer profiling
* Streaming fraud detection
* Adaptive threshold selection
* Feature importance analysis
* Model comparison (LOF, One-Class SVM, ECOD)
* SHAP explanation for anomaly interpretation

---

# License

This project is intended for educational, research, and portfolio purposes.

The synthetic dataset does not contain any real customer information.
