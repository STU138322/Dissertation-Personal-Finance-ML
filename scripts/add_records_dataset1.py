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

print("\n=== Generating Enhanced Synthetic Records for Dataset1 ===")
df = pd.read_csv(RAW_PATH, parse_dates=['Date'])

print(f"Original record count: {df.shape[0]}")
target_rows = 1500
records_to_add = target_rows - df.shape[0]

if records_to_add <= 0:
    print("No synthetic records needed.")
else:
    print(f"Adding {records_to_add} synthetic records...")

    # Expanded category coverage
    expense_categories = [
        "Food", "Groceries", "Rent", "Transport", "Entertainment",
        "Utilities", "Medical", "Insurance", "Subscriptions", "Education",
        "Pets", "Debt", "Health"
    ]
    income_categories = ["Salary", "Bonus", "Investment", "Gift", "Tax Return"]

    additional_data = []

    # Date range
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=90)
    end_date = today

    # Target distribution buckets
    n_expense = int(records_to_add * 0.75)
    n_income = int(records_to_add * 0.20)
    n_edge = int(records_to_add * 0.05)

    # Normal expenses
    for _ in range(n_expense):
        date = fake.date_between(start_date=start_date, end_date=end_date)
        category = random.choice(expense_categories)
        amount = round(random.triangular(14, 1800, 1000), 2)
        amount += random.uniform(-20, 20)
        amount = max(amount, 5)
        additional_data.append({
            'Date': date,
            'Category': category,
            'Category Type': "Expense",
            'Amount': amount,
            'Source': 'synthetic'
        })

    # Normal incomes
    for _ in range(n_income):
        date = fake.date_between(start_date=start_date, end_date=end_date)
        category = random.choice(income_categories)
        amount = round(random.triangular(500, 6000, 2500), 2)
        amount += random.uniform(-50, 50)
        amount = max(amount, 100)
        additional_data.append({
            'Date': date,
            'Category': category,
            'Category Type': "Income",
            'Amount': amount,
            'Source': 'synthetic'
        })

    # Moderate positive savings periods (Savings_Rate ~ 1.5–4)
    for _ in range(10):
        date = fake.date_between(start_date=start_date, end_date=end_date)
        additional_data.extend([
            {
                'Date': date,
                'Category': 'Investment',
                'Category Type': "Income",
                'Amount': round(random.uniform(500, 800), 2),
                'Source': 'synthetic'
            },
            {
                'Date': date,
                'Category': 'Subscription',
                'Category Type': "Expense",
                'Amount': round(random.uniform(50, 150), 2),
                'Source': 'synthetic'
            }
        ])

    # Bounded negative savings (Savings_Rate ~ –3 to –1)
    for _ in range(10):
        date = fake.date_between(start_date=start_date, end_date=end_date)
        additional_data.extend([
            {
                'Date': date,
                'Category': 'Salary',
                'Category Type': "Income",
                'Amount': round(random.uniform(300, 600), 2),
                'Source': 'synthetic'
            },
            {
                'Date': date,
                'Category': 'Medical',
                'Category Type': "Expense",
                'Amount': round(random.uniform(900, 1800), 2),
                'Source': 'synthetic'
            }
        ])

    # Combine
    df_synthetic = pd.DataFrame(additional_data)
    df = pd.concat([df, df_synthetic], ignore_index=True)

    # Final processing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['Source'] = df['Source'].fillna('original')

    # Force all amounts positive to match convention
    df['Amount'] = df['Amount'].abs()

    # Save
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Updated Dataset1 saved to {PROCESSED_PATH}")
    print(f"Total record count: {df.shape[0]}")
