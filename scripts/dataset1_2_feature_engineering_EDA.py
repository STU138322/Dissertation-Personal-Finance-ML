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
    df['Category Type'] = df['Category Type'].str.title()
    df['Category'] = df['Category'].str.title()

    # Safely handle Source column
    if 'Source' not in df.columns:
        if name == 'dataset1':
            df['Source'] = 'original'  # Only dataset1 needs it
        else:
            pass  # dataset2 has no Source, no need to add
    else:
        df['Source'] = df['Source'].fillna('original')

    # Create Income and Expense fields based on dataset
    if name == 'dataset1':
        df['Income'] = df['Amount'].where(df['Amount'] > 0, 0)
        df['Expense'] = df['Amount'].where(df['Amount'] < 0, 0)
    else:
        df['Income'] = df['Amount'].where(df['Category Type'] == 'Income', 0)
        df['Expense'] = df['Amount'].where(df['Category Type'] == 'Expense', 0)

    # Set index for resampling
    df = df.set_index('Date')

    # Resample separately
    resampled = df[['Income', 'Expense']].resample('3D').sum().reset_index()

    # Drop periods with no financial activity (after resampling)
    resampled = resampled[(resampled['Income'] != 0) | (resampled['Expense'] != 0)]

    # Recreate Net Savings and features
    resampled['Net_Savings'] = resampled['Income'] + resampled['Expense']
    resampled['Savings_Rate'] = resampled['Net_Savings'] / resampled['Income'].replace(0, np.nan)

    resampled['Rolling_Income'] = resampled['Income'].rolling(window=3, min_periods=1).mean()
    resampled['Rolling_Expense'] = resampled['Expense'].rolling(window=3, min_periods=1).mean()
    resampled['Rolling_Savings'] = resampled['Net_Savings'].rolling(window=3, min_periods=1).mean()

    # Keep Source, Category, and Category Type
    df['Category'] = df['Category'].fillna('Unknown')
    df['Category Type'] = df['Category Type'].fillna('Unknown')

    if 'Source' in df.columns:
        df['Source'] = df['Source'].fillna('original')

    # Begin EDA
    os.makedirs(f'{NOTEBOOKS_DIR}/{name}', exist_ok=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = resampled[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']].corr()
    sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    fmt='.2f'
    )
    plt.title(f'Correlation Heatmap - {name}')
    plt.tight_layout()
    plt.savefig(f'{NOTEBOOKS_DIR}/{name}/correlation_heatmap.png')
    plt.close()

    # Distribution Plots
    for col in ['Income', 'Expense', 'Net_Savings', 'Savings_Rate']:
        plt.figure(figsize=(8, 4))
        sns.histplot(resampled[col], kde=True)
        plt.title(f'{col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/{col}_distribution.png')
        plt.close()

    # Skewness Visualization
    skewness = resampled[['Income', 'Expense', 'Net_Savings']].skew()
    print(f"Skewness for {name}:\n{skewness}\n")

    # Log1p plots for positive fields
    for col in ['Income', 'Net_Savings']:
        plt.figure(figsize=(8, 4))
        sns.histplot(np.log1p(resampled[col]), kde=True)
        plt.title(f'Log1p {col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/log1p_{col}_distribution.png')
        plt.close()

    # Special handling for Expense (take abs before log1p)
    plt.figure(figsize=(8, 4))
    sns.histplot(np.log1p(np.abs(resampled['Expense'])), kde=True)
    plt.title(f'Log1p Absolute Expense Distribution - {name}')
    plt.tight_layout()
    plt.savefig(f'{NOTEBOOKS_DIR}/{name}/log1p_Expense_distribution.png')
    plt.close()

    # Save engineered dataset only once after full processing
    resampled.to_csv(f'{ENGINEERED_DIR}/engineered_{name}.csv', index=False)
    print(f"Updated and saved: engineered_{name}.csv")

print("\nFeature Engineering + EDA complete for Phase 1&2.")