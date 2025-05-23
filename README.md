# Dissertation – Personal Finance Forecasting with AI

STU138322 – Final Year BSc Computing Dissertation  
Project Title: Leveraging AI and Databases for Predictive Analysis in Personal Finance

## Project Overview

This dissertation explores how AI models and structured databases can be used to predict the growth of personal savings based on historical income, expenses, and spending behavior. It includes a full machine learning pipeline along with an interactive dashboard for visualizing financial trends, model performance, and exploratory data analysis (EDA).

## Project Structure

```
.
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
├── notebooks/               # Exploratory analysis and hypothesis testing
│ ├── Hypothesis_tests/      
│ └── Phase1&2/              # All EDA charts for engineered datasets
│
├── outputs/ # Evaluation metrics, plots, and predictions
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
├── dashboard.py             # Combined Streamlit dashboard (EDA + Models)
├── main.py                  # Runs model training, testing, benchmarks, growth tracking
├── requirements.txt
└── README.md
```

## Setup Instructions

pip install -r requirements.txt

# Clean, synthesize, and engineer data;
python main.py --clean --synthesize --feature

# Run hypothesis testing;
python main.py --hypothesis

# Train and evaluate models:
python main.py --train --blindtest

# Generate benchmark, growth, and summary charts:
python main.py --growth --summary

# Launch dashboard:
python main.py --dashboard


## Models Used

- Linear Regression
- Random Forest
- Support Vector Regression (SVR)

## Evaluation Metrics

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² – Coefficient of Determination
- Segment-level metrics:
    - All Data
    - Original Only
    - Synthetic Only

## Visualization Highlights

- Dataset Explorer:
    - KDE distributions for raw and log-transformed features
    - Shapiro-Wilk p-values pre/post log1p
    - Category vs Encoded Category + Legend
- Growth Tracking:
    - Actual vs Predicted Savings Rate
    - Filter by time range and source (original/synthetic)
- Cross-Validation Metrics:
    - 5-Fold scores with line charts
- Model Metric Summary:
    - MAE, RMSE, R² per model/dataset
    - Best performers auto-highlighted
    - Summary chart saved as .png
- User Upload (Placeholder):
    - Allows .csv preview (integration possible planned)

## Additional Ouputs

- model_comparison_summary.csv – All metrics across models and datasets
- model_comparison_summary.txt – Highlighted best MAE/RMSE/R²
- model_comparison_chart.png – Visualized bar chart for summary metrics
- growth_tracker folders – Track actual vs predicted savings over time
- metrics.txt, cv_metrics.txt, predictions.csv – Per-model logs

## Author

Sander  
Student ID: STU138322  
Final Year BSc Computing  
Arden University