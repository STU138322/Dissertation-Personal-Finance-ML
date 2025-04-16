import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

file_path = 'data/engineered/engineered_dataset2.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

print("\n-- Engineered Dataset 2 Info --")
print(df.info())

print("\n-- Descriptive Statistics --")
print(df.describe())

os.makedirs('notebooks', exist_ok=True)

plt.figure(figsize=(10, 4))
sns.lineplot(data=df, x='Date', y='Net_Savings')
plt.title('Net Savings Over Time (Dataset 2)')
plt.tight_layout()
plt.savefig('notebooks/engineered_EDA/eda_engineered_dataset2_netsavings.png')
plt.show()

plt.figure(figsize=(10, 4))
sns.lineplot(data=df, x='Date', y='Rolling_Savings')
plt.title('Rolling 3-Month Net Savings (Dataset 2)')
plt.tight_layout()
plt.savefig('notebooks/engineered_EDA/eda_engineered_dataset2_rolling_savings.png')
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['Savings_Rate'], kde=True)
plt.title('Savings Rate Distribution (Dataset 2)')
plt.tight_layout()
plt.savefig('notebooks/engineered_EDA/eda_engineered_dataset2_savings_rate_dist.png')
plt.show()

print("\nEDA on engineered Dataset 2 complete. Charts saved in /notebooks/engineered_EDA/")
