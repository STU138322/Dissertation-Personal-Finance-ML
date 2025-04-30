import pandas as pd
import sqlite3

# Path to the shared SQLite database
DB_PATH = 'db/finance_data.db'

# Available tables
TABLE_TRAIN = 'savings_data_train'
TABLE_BLINDTEST = 'savings_data_blindtest'

# Targeted fields (now includes encoded category)
FEATURES = ['Income', 'Expense', 'Rolling_Income', 'Rolling_Expense', 'Category_Encoded']
TARGET = 'Savings_Rate'

def load_data(table_name):
    """
    Load dataset from the SQLite database.

    Parameters:
    - table_name (str): Name of the table to load (e.g., 'savings_data_train')

    Returns:
    - DataFrame with features and target columns.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
