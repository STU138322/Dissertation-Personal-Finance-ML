import os
import pandas as pd
import sqlite3

# Paths
train_csv_path = 'data/engineered/engineered_dataset2.csv'
blind_csv_path = 'data/engineered/engineered_dataset1.csv'
db_path = 'db/finance_data.db'

# Table names
table_train = 'savings_data_train'
table_blindtest = 'savings_data_blindtest'

# Create db directory if it doesn't exist
os.makedirs('db', exist_ok=True)

# Connect to SQLite database (or create if not exists)
print(f"Connecting to database at: {db_path}")
conn = sqlite3.connect(db_path)

# Load and save training dataset
print("Loading training CSV data...")
df_train = pd.read_csv(train_csv_path, parse_dates=['Date'])
df_train.to_sql(table_train, conn, if_exists='replace', index=False)
print(f"Table '{table_train}' created/updated in {db_path}")

# Load and save blindtest dataset
print("Loading blindtest CSV data...")
df_blind = pd.read_csv(blind_csv_path, parse_dates=['Date'])
df_blind.to_sql(table_blindtest, conn, if_exists='replace', index=False)
print(f"Table '{table_blindtest}' created/updated in {db_path}")

# Optional: preview 3 rows from each table
print("\nSample rows from training table:")
print(pd.read_sql_query(f"SELECT * FROM {table_train} LIMIT 3", conn))

print("\nSample rows from blindtest table:")
print(pd.read_sql_query(f"SELECT * FROM {table_blindtest} LIMIT 3", conn))

# Close the connection
conn.close()
print("Database connection closed.")
