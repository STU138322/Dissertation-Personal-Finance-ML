import pandas as pd
import numpy as np
import os
from faker import Faker
import random

# Init faker and random seed for reproducibility
fake = Faker()
random.seed(42)
pd.set_option('display.max_columns', None)

RAW_PATH = 'data/processed/cleaned_dataset1.csv'
PROCESSED_PATH = 'data/processed/cleaned_dataset1.csv'

print("Loading dataset...")
df = pd.read_csv(RAW_PATH, parse_dates=['Date'])

print("Adding synthetic records to match Dataset 2 scale...")
existing_rows = df.shape[0]
target_rows = 1500
records_to_add = target_rows - existing_rows

if records_to_add > 0:
    additional_data = []
    categories = df['Category'].dropna().unique().tolist()
    types = df['Category Type'].dropna().unique().tolist()

    for _ in range(records_to_add):
        date = fake.date_between(start_date='-4y', end_date='today')
        category = random.choice(categories)
        category_type = random.choice(types)
        amount = round(random.uniform(50, 3000), 2)
        if category_type == 'expense':
            amount = -abs(amount)

        additional_data.append({
            'Date': date,
            'Category': category,
            'Category Type': category_type,
            'Amount': amount,
            'Source': 'synthetic'
        })

    df_new = pd.DataFrame(additional_data)
    df = pd.concat([df, df_new], ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values(by='Date')

df['Source'] = df['Source'].fillna('original')

os.makedirs('data/processed', exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
print(f"Updated dataset with synthetic records saved to {PROCESSED_PATH}")
print("Final shape:", df.shape)
