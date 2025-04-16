import pandas as pd
import numpy as np
import os

# Load cleaned dataset with Source column
df = pd.read_csv('data/processed/cleaned_dataset1.csv', parse_dates=['Date'])

# Sort by date
df = df.sort_values('Date')

# Extract year and month for clarity (optional)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Normalize category values
df['Category Type'] = df['Category Type'].str.lower()

# Create Income and Expense columns
df['Income'] = df['Amount'].where(df['Category Type'] == 'income', 0)
df['Expense'] = df['Amount'].where(df['Category Type'] == 'expense', 0)

# Group by month and Source so we keep both synthetic and original
monthly_summary = df.groupby([pd.Grouper(key='Date', freq='M'), 'Source']).agg({
    'Income': 'sum',
    'Expense': 'sum'
}).reset_index()

# Feature engineering
monthly_summary['Net_Savings'] = monthly_summary['Income'] + monthly_summary['Expense']
monthly_summary['Savings_Rate'] = monthly_summary['Net_Savings'] / monthly_summary['Income'].replace(0, np.nan)

# Rolling features within each Source group
monthly_summary['Rolling_Income'] = (
    monthly_summary.groupby('Source')['Income']
    .rolling(window=3, min_periods=1).mean().reset_index(drop=True)
)
monthly_summary['Rolling_Expense'] = (
    monthly_summary.groupby('Source')['Expense']
    .rolling(window=3, min_periods=1).mean().reset_index(drop=True)
)
monthly_summary['Rolling_Savings'] = (
    monthly_summary.groupby('Source')['Net_Savings']
    .rolling(window=3, min_periods=1).mean().reset_index(drop=True)
)

# Final clean-up
monthly_summary = monthly_summary.fillna(0)

# Save output
os.makedirs('data/engineered', exist_ok=True)
monthly_summary.to_csv('data/engineered/engineered_dataset1.csv', index=False)

print("Feature engineering complete. Both 'synthetic' and 'original' data preserved.")
