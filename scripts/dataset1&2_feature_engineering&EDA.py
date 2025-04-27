# Phase 1 & 2 Feature Engineering and EDA Update

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")

# Paths
RAW_DIR = 'data/processed'
ENGINEERED_DIR = 'data/engineered'
NOTEBOOKS_DIR = 'notebooks/phase1&2'
os.makedirs(ENGINEERED_DIR, exist_ok=True)
os.makedirs(f'{NOTEBOOKS_DIR}/dataset1', exist_ok=True)
os.makedirs(f'{NOTEBOOKS_DIR}/dataset2', exist_ok=True)

# Load cleaned datasets
datasets = {
    'dataset1': pd.read_csv(f'{RAW_DIR}/cleaned_dataset1.csv', parse_dates=['Date']),
    'dataset2': pd.read_csv(f'{RAW_DIR}/cleaned_dataset2.csv', parse_dates=['Date'])
}

for name, df in datasets.items():
    print(f"\nProcessing {name}...")

    # Sort and basic fixes
    df = df.sort_values('Date')
    df['Category Type'] = df['Category Type'].str.lower()
    df['Category'] = df['Category'].str.title()
    df['Source'] = df['Source'].fillna('original')

    # Create Income and Expense fields
    df['Income'] = df['Amount'].where(df['Category Type'] == 'income', 0)
    df['Expense'] = df['Amount'].where(df['Category Type'] == 'expense', 0)

    # Resample to Daily or 3-hourly to hit >1000 records
    df = df.set_index('Date').resample('3H').sum().reset_index()

    # Fill missing values
    df['Income'] = df['Income'].fillna(0)
    df['Expense'] = df['Expense'].fillna(0)

    # Recreate Net Savings and features
    df['Net_Savings'] = df['Income'] + df['Expense']
    df['Savings_Rate'] = df['Net_Savings'] / df['Income'].replace(0, np.nan)

    df['Rolling_Income'] = df['Income'].rolling(window=3, min_periods=1).mean()
    df['Rolling_Expense'] = df['Expense'].rolling(window=3, min_periods=1).mean()
    df['Rolling_Savings'] = df['Net_Savings'].rolling(window=3, min_periods=1).mean()

    # Keep Source, Category, and Category Type
    df['Category'] = df['Category'].fillna('Unknown')
    df['Category Type'] = df['Category Type'].fillna('Unknown')
    df['Source'] = df['Source'].fillna('original')

    # Save engineered dataset
    engineered_path = f'{ENGINEERED_DIR}/engineered_{name}.csv'
    df.to_csv(engineered_path, index=False)
    print(f"Saved {engineered_path} ({df.shape[0]} records)")

    # Begin EDA
    os.makedirs(f'{NOTEBOOKS_DIR}/{name}', exist_ok=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap - {name}')
    plt.tight_layout()
    plt.savefig(f'{NOTEBOOKS_DIR}/{name}/correlation_heatmap.png')
    plt.close()

    # Distribution Plots
    for col in ['Income', 'Expense', 'Net_Savings', 'Savings_Rate']:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/{col}_distribution.png')
        plt.close()

    # Skewness Visualization
    skewness = df[['Income', 'Expense', 'Net_Savings']].skew()
    print(f"Skewness for {name}:\n{skewness}\n")
    
    for col in ['Income', 'Expense', 'Net_Savings']:
        plt.figure(figsize=(8, 4))
        sns.histplot(np.log1p(df[col]), kde=True)
        plt.title(f'Log1p {col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/log1p_{col}_distribution.png')
        plt.close()

    # Label Encoding
    encoders = {}
    for label_col in ['Category', 'Category Type']:
        encoder = LabelEncoder()
        df[label_col + '_Encoded'] = encoder.fit_transform(df[label_col].astype(str))
        encoders[label_col] = encoder

    df.to_csv(f'{ENGINEERED_DIR}/engineered_{name}.csv', index=False)
    print(f"Updated with Label Encoded fields. Final saved: engineered_{name}.csv")

print("\nFeature Engineering + EDA complete for Phase 1&2.")