# Dissertation – Personal Finance Forecasting with AI

STU138322 – Final Year BSc Computing Dissertation  
Project Title: Leveraging AI and Databases for Predictive Analysis in Personal Finance

## Project Overview

This project demonstrates the use of AI models and databases to predict personal net savings growth based on individual financial history (income, expenses, and savings behavior). It features a full pipeline for data cleaning, augmentation, model training, benchmarking, and performance visualization.

## Project Structure

```
.
├── benchmarks/              # Threshold summaries and visuals
│ ├── linear_regression/
│ ├── linear_regression_blindtest/
│ ├── random_forest/
│ ├── random_forest_blindtest/
│ ├── svr/
│ └── svr_blindtest/
│
├── data/                    # Raw, cleaned, and engineered datasets
│ ├── raw/
│ ├── cleaned/
│ └── engineered/
│
├── db/                      # SQLite database with financial records
│
├── models/                  # Trained model files (.pkl)
│ ├── linear_regression/
│ ├── random_forest/
│ └── svr/
│
├── notebooks/               # EDA charts for processed and engineered datasets
│ ├── Hypothesis_tests/      
│ └── Phase1&2/              # All EDA charts for engineered datasets
│
├── outputs/ # Model metrics, predictions, benchmarks
│ ├── charts_all_data/
│ ├── charts_original_only/
│ ├── charts_synthetic_only/
│ ├── linear_regression/
│ ├── linear_regression_blindtest/
│ ├── random_forest/
│ ├── random_forest_blindtest/
│ ├── svr/
│ └── svr_blindtest/
│
├── predicted_savings_tracking/ # Growth tracker results (Dataset2)
├── predicted_savings_tracking_blindtest/ # Growth tracker results (Dataset1)
│
├── scripts/                 # All modular Python scripts for pipeline stages
│
├── models_datasets.py # Streamlit dashboard for dataset1 & 2 Exploratory Data Analysis (EDA)
├── models_dashboard.py # Streamlit dashboard for growth, benchmarks, metrics
├── data_pipeline.py         # Runs data prep: clean → augment → engineer → DB
├── main.py                  # Runs model training, testing, benchmarks, growth tracking
├── report_export.py         # Packages all outputs for submission
├── requirements.txt
└── README.md
```

## Setup Instructions

pip install -r requirements.txt

# Run full data prep
python data_pipeline.py --all

# Train, evaluate, benchmark and track
python main.py --train --blindtest --benchmarks --growth --summary

# Launch dashboards:
python main.py --dashboard-datasets
python main.py --dashboard-models

# Export results
python report_export.py

## Models Used

- Linear Regression
- Random Forest
- Support Vector Regression (SVR)

## Evaluation Metrics

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² – Coefficient of Determination
- Threshold Accuracy – % of predictions ≥ 50%, 100%, 150% of true value

## Visualization Highlights

- Growth tracking over time by model and user
- Benchmark accuracy (50/100/150%)
- Model comparison by MAE, RMSE, R²
- Category/source-type breakdown (original vs synthetic)
- Blindtest vs training comparison graphs

## Author

Sander  
Student ID: STU138322  
Final Year BSc Computing  
Arden University