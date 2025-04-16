import pandas as pd
import numpy as np
import os

# Load cleaned dataset with Source column
df = pd.read_csv('data/processed/cleaned_dataset1.csv', parse_dates=['Date'])

# Ensure correct sort order
df = df.sort_values('Date')

# Extract year and month (optional, for reference)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Normalize category values
df['Category Type'] = df['Category Type'].str.lower()

# Separate income and expense
df['Income'] = df['Amount'].where(df['Category Type'] == 'income', 0)
df['Expense'] = df['Amount'].where(df['Category Type'] == 'expense', 0)

# Group by month-end and calculate aggregates
monthly_summary = df.groupby(pd.Grouper(key='Date', freq='M')).agg({
    'Income': 'sum',
    'Expense': 'sum',
    'Source': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
}).reset_index()

# Feature engineering
monthly_summary['Net_Savings'] = monthly_summary['Income'] + monthly_summary['Expense']  # Expense is negative
monthly_summary['Savings_Rate'] = monthly_summary['Net_Savings'] / monthly_summary['Income'].replace(0, np.nan)
monthly_summary['Rolling_Income'] = monthly_summary['Income'].rolling(window=3).mean()
monthly_summary['Rolling_Expense'] = monthly_summary['Expense'].rolling(window=3).mean()
monthly_summary['Rolling_Savings'] = monthly_summary['Net_Savings'].rolling(window=3).mean()

# Fill any resulting NaNs
monthly_summary = monthly_summary.fillna(0)

# Save to engineered dataset
os.makedirs('data/engineered', exist_ok=True)
monthly_summary.to_csv('data/engineered/engineered_dataset1.csv', index=False)

print("Feature engineering complete for Dataset 1. Output saved to data/engineered/engineered_dataset1.csv")