import pandas as pd
import numpy as np
import os
import random
from faker import Faker

# Init
fake = Faker()
random.seed(42)

# Paths
RAW_PATH = 'data/processed/cleaned_dataset1.csv'
PROCESSED_PATH = 'data/processed/cleaned_dataset1.csv'

print("\n=== Generating Realistic Synthetic Dataset1 (1600 Total) ===")
df = pd.read_csv(RAW_PATH, parse_dates=['Date'])

# Add 'Source' column to mark all existing records as original
df['Source'] = 'original'

print(f"Original record count: {df.shape[0]}")
target_rows = 1600
records_to_add = target_rows - df.shape[0]

if records_to_add <= 0:
    print("No synthetic records needed.")
else:
    print(f"Adding {records_to_add} synthetic records...")

    income_categories = ["Salary", "Bonus", "Investment", "Gift", "Tax Return"]
    expense_categories = [
        "Food", "Groceries", "Rent", "Transport", "Entertainment",
        "Utilities", "Medical", "Insurance", "Subscriptions", "Education",
        "Pets", "Debt", "Health"
    ]

    # Start from the day after the latest original record
    last_date = df['Date'].max()
    start_date = last_date + pd.Timedelta(days=1)

    additional_data = []
    half = records_to_add // 2

    for i in range(half):
        base_date = start_date + pd.Timedelta(days=random.randint(i * 3, i * 5))

        income_date = base_date + pd.Timedelta(days=random.choice([0, 1, 2]))
        expense_date = base_date + pd.Timedelta(days=random.choice([3, 4, 5]))

        income_amt = max(round(abs(random.gauss(2900, 800)) + random.uniform(300, 800), 2), 500)
        expense_amt = max(round(abs(random.gauss(1500, 600)) + random.uniform(300, 700), 2), 500)

        # Every 5th record: income-only or expense-only
        if i % 5 == 0:
            if random.random() < 0.5:
                additional_data.append({
                    'Date': income_date,
                    'Category': random.choice(income_categories),
                    'Category Type': "Income",
                    'Amount': income_amt,
                    'Source': 'synthetic'
                })
            else:
                additional_data.append({
                    'Date': expense_date,
                    'Category': random.choice(expense_categories),
                    'Category Type': "Expense",
                    'Amount': expense_amt,
                    'Source': 'synthetic'
                })
            continue

        # Regular income + expense pair
        additional_data.append({
            'Date': income_date,
            'Category': random.choice(income_categories),
            'Category Type': "Income",
            'Amount': income_amt,
            'Source': 'synthetic'
        })

        additional_data.append({
            'Date': expense_date,
            'Category': random.choice(expense_categories),
            'Category Type': "Expense",
            'Amount': expense_amt,
            'Source': 'synthetic'
        })

    # Finalize
    df_synthetic = pd.DataFrame(additional_data)
    df = pd.concat([df, df_synthetic], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['Amount'] = df['Amount'].abs()

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Updated Dataset1 saved to {PROCESSED_PATH}")
    print(f"Total record count: {df.shape[0]}")
