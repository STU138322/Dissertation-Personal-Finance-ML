import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

file_path = 'data/processed/cleaned_dataset2.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

print("\n-- Dataset Information --")
print(df.info())

print("\n-- Descriptive Statistics --")
print(df.describe())

print("\n-- First Couple of Rows --")
print(df.head())

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

monthly_total = df.groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum()

plt.figure(figsize=(10, 4))
monthly_total.plot()
plt.title('Total Transaction Amount per Month')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.tight_layout()
os.makedirs('notebooks', exist_ok=True)
plt.savefig('notebooks/processed_EDA/eda_dataset2_monthly_total.png')
plt.show()

if 'Category Type' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Category Type')
    plt.title('Transaction Type Count')
    plt.tight_layout()
    plt.savefig('notebooks/processed_EDA/eda_dataset2_type_count.png')
    plt.show()

if 'Category' in df.columns:
    plt.figure(figsize=(8, 6))
    df['Category'].value_counts().head(10).plot(kind='barh')
    plt.title('Top 10 Transaction Categories')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('notebooks/processed_EDA/eda_dataset2_top_categories.png')
    plt.show()

if 'Category Type' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Category Type', y='Amount')
    plt.title('Transaction Amount Distribution by Type')
    plt.tight_layout()
    plt.savefig('notebooks/processed_EDA/eda_dataset2_amount_boxplot.png')
    plt.show()

print("\nEDA for Dataset 2 complete. Charts saved in /notebooks/processed_EDA/")