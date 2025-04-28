import pandas as pd
import numpy as np
import os
from faker import Faker
import random

# Init faker and random seed for reproducibility
fake = Faker()
random.seed(42)
pd.set_option('display.max_columns', None)

# Paths
RAW_PATH = 'data/processed/cleaned_dataset1.csv'
PROCESSED_PATH = 'data/processed/cleaned_dataset1.csv'

print("\n=== Adding Synthetic Records to Dataset1 ===")
print("Loading cleaned dataset...")
df = pd.read_csv(RAW_PATH, parse_dates=['Date'])

# Check initial stats
print(f"Original record count: {df.shape[0]}")

# Target size
target_rows = 1500
records_to_add = target_rows - df.shape[0]

# Check if we actually need to add
if records_to_add <= 0:
    print(f"No synthetic records needed. Already {df.shape[0]} records.")
else:
    print(f"Adding {records_to_add} synthetic records to reach {target_rows} total...")

    # Prepare category lists
    categories = df['Category'].dropna().unique().tolist()
    category_types = df['Category Type'].dropna().unique().tolist()

    additional_data = []

    for _ in range(records_to_add):
        date = fake.date_between(start_date='-3y', end_date='today')

        # Randomly assign category and type
        category = random.choice(categories)
        category_type = random.choice(category_types)

        # Generate amount based on category type
        if category_type == 'Income':
            amount = round(random.uniform(200, 5000), 2)  # realistic income
        else:
            amount = -round(random.uniform(20, 3000), 2)   # realistic expenses (negative)

        record = {
            'Date': date,
            'Category': category,
            'Category Type': category_type,
            'Amount': amount,
            'Source': 'synthetic'
        }
        additional_data.append(record)

    # Create DataFrame and merge
    df_synthetic = pd.DataFrame(additional_data)
    df = pd.concat([df, df_synthetic], ignore_index=True)

# Final cleaning
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')
df['Source'] = df['Source'].fillna('original')

# Save updated dataset
os.makedirs('data/processed', exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print(f"\nUpdated dataset saved to {PROCESSED_PATH}")
print(f"Final record count: {df.shape[0]}")
print("=== Synthetic Record Addition Complete ===")
