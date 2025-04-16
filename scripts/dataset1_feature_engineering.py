import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/processed/cleaned_dataset1.csv', parse_dates=['Date'])

df = df.sort_values('Date')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df['Category Type'] = df['Category Type'].str.lower()

df['Income'] = df['Amount'].where(df['Category Type'] == 'income', 0)
df['Expense'] = df['Amount'].where(df['Category Type'] == 'expense', 0)

monthly_summary = df.groupby([pd.Grouper(key='Date', freq='ME')]).agg({
    'Income': 'sum',
    'Expense': 'sum'
}).reset_index()

monthly_summary['Net_Savings'] = monthly_summary['Income'] + monthly_summary['Expense']  # Expense is negative
monthly_summary['Savings_Rate'] = monthly_summary['Net_Savings'] / monthly_summary['Income'].replace(0, np.nan)

monthly_summary['Rolling_Income'] = monthly_summary['Income'].rolling(window=3).mean()
monthly_summary['Rolling_Expense'] = monthly_summary['Expense'].rolling(window=3).mean()
monthly_summary['Rolling_Savings'] = monthly_summary['Net_Savings'].rolling(window=3).mean()

monthly_summary = monthly_summary.fillna(0)

os.makedirs('data/engineered', exist_ok=True)
monthly_summary.to_csv('data/engineered/engineered_dataset1.csv', index=False)

print("Feature engineering complete for Dataset 1. Output saved to data/engineered/engineered_dataset1.csv")