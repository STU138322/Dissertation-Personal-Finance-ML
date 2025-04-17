# Dissertation – Personal Finance Forecasting with AI

STU138322 – Final Year BSc Computing Dissertation  
Project Title: Leveraging AI and Databases for Predictive Analysis in Personal Finance

## Project Overview

This project demonstrates the use of AI models and databases to predict personal net savings growth based on individual financial history (income, expenses, and savings behavior). It features a full pipeline for data cleaning, augmentation, model training, benchmarking, and performance visualization.

## Project Structure

.
├── data/                  # Raw, cleaned, and engineered datasets
├── db/                    # SQLite database with financial records
├── models/                # Trained model files (pkl)
├── notebooks/             # EDA charts for processed and engineered datasets
├── outputs/               # Metrics, predictions, threshold charts
├── benchmarks/            # Threshold summaries and visuals
├── scripts/               # All modular Python scripts for pipeline stages
├── submission_export/     # Final collected results for report appendices
├── main.py                # Runs model training, testing, benchmarks, growth tracking
├── data_pipeline.py       # Runs data prep: clean → augment → engineer → DB
├── report_export.py       # Packages all outputs for submission
├── requirements.txt
└── README.md

## Setup Instructions

pip install -r requirements.txt

# Run full data prep
python data_pipeline.py --all

# Train, evaluate, benchmark and track
python main.py --train --blindtest --benchmarks --growth --summary

# Export results
python report_export.py

## Models Used

- Linear Regression
- Decision Tree
- Random Forest

## Evaluation Metrics

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² – Coefficient of Determination
- Savings Growth Threshold Accuracy (≥ 50%, ≥ 100%, ≥ 150%)

## Visualization Highlights

- Benchmark thresholds (50/100/150%)
- Time-based growth prediction tracking (per user)
- Full performance comparison charts across models and datasets

## Author

Sander  
Student ID: STU138322  
Final Year BSc Computing  
Arden University