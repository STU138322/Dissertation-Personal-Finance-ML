import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

    df = df.sort_values('Date')

    # Normalize and standardize category types
    df['Category Type'] = df['Category Type'].astype(str).str.strip().str.lower()
    standard_types = {
        'expenses': 'expense',
        'expense': 'expense',
        'income': 'income',
        'savings': 'savings',
        'saving': 'savings'
    }
    df['Category Type'] = df['Category Type'].map(lambda x: standard_types.get(x, 'unknown'))
    df['Category'] = df['Category'].astype(str).str.strip().str.title()

    df['Source'] = df.get('Source', 'original')

    # Feature engineering for income/expense
    df['Income'] = df.apply(
        lambda row: abs(row['Amount']) if row['Category Type'] in ['income', 'savings'] else 0, axis=1
    )
    df['Expense'] = df.apply(
        lambda row: -abs(row['Amount']) if row['Category Type'] == 'expense' else 0, axis=1
    )

    df = df.set_index('Date')

    # Resample every 2 days
    resampled = df[['Income', 'Expense']].resample('2D').sum().reset_index()
    resampled = resampled[(resampled['Income'] != 0) | (resampled['Expense'] != 0)]

    # Feature Engineering
    resampled['Net_Savings'] = resampled['Income'] + resampled['Expense']
    resampled['Savings_Rate'] = resampled['Net_Savings'] / resampled['Income'].replace(0, np.nan)
    resampled['Rolling_Income'] = resampled['Income'].rolling(window=3, min_periods=1).mean()
    resampled['Rolling_Expense'] = resampled['Expense'].rolling(window=3, min_periods=1).mean()
    resampled['Rolling_Savings'] = resampled['Net_Savings'].rolling(window=3, min_periods=1).mean()

    # Most common category per 2D period
    dom_cat = df['Category'].resample('2D').agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    dom_source = df['Source'].resample('2D').agg(lambda x: 'synthetic' if 'synthetic' in x.values else 'original').reset_index()

    resampled = pd.merge(resampled, dom_cat, on='Date', how='left')
    resampled = pd.merge(resampled, dom_source, on='Date', how='left')

    # Encode
    if 'Category' in resampled.columns:
        le = LabelEncoder()
        resampled['Category_Encoded'] = le.fit_transform(resampled['Category'].fillna('Unknown'))
    else:
        resampled['Category_Encoded'] = 0

    # Debug: Non-zero Expense Periods
    non_zero_expense = resampled[resampled['Expense'] != 0]
    print(f"Non-zero expense periods in {name}: {len(non_zero_expense)}")
    non_zero_expense.to_csv(f'{ENGINEERED_DIR}/debug_expense_periods_{name}.csv', index=False)

    print(f"\nResampled {name} preview:")
    print(resampled[['Date', 'Income', 'Expense', 'Net_Savings', 'Savings_Rate']].head())

    # === EDA Visuals ===
    os.makedirs(f'{NOTEBOOKS_DIR}/{name}', exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = resampled[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title(f'Correlation Heatmap - {name}')
    plt.tight_layout()
    plt.savefig(f'{NOTEBOOKS_DIR}/{name}/correlation_heatmap.png')
    plt.close()

    # Distributions
    for col in ['Income', 'Expense', 'Net_Savings', 'Savings_Rate']:
        plt.figure(figsize=(8, 4))
        sns.histplot(resampled[col].dropna(), kde=True)
        plt.title(f'{col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/{col}_distribution.png')
        plt.close()

    for col in ['Income', 'Net_Savings']:
        plt.figure(figsize=(8, 4))
        safe_col = resampled[col].where(resampled[col] >= 0)
        sns.histplot(np.log1p(safe_col.dropna()), kde=True)
        plt.title(f'Log1p {col} Distribution - {name}')
        plt.tight_layout()
        plt.savefig(f'{NOTEBOOKS_DIR}/{name}/log1p_{col}_distribution.png')
        plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(np.log1p(np.abs(resampled['Expense'])), kde=True)
    plt.title(f'Log1p Absolute Expense Distribution - {name}')
    plt.tight_layout()
    plt.savefig(f'{NOTEBOOKS_DIR}/{name}/log1p_Expense_distribution.png')
    plt.close()

    resampled.to_csv(f'{ENGINEERED_DIR}/engineered_{name}.csv', index=False)
    print(f"Saved: engineered_{name}.csv")

print("\nFeature Engineering + EDA complete for both datasets.")
