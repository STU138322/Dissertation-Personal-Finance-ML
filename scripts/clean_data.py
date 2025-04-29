import pandas as pd
import os

# Paths
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# List of raw input files
files = [
    ('Personal_Finance_Dataset1.csv', 'cleaned_dataset1.csv'),
    ('Personal_Finance_Dataset2.csv', 'cleaned_dataset2.csv')
]

def CleaningFunction(df, dataset_name):
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=['Date', 'Amount'])

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount'])

    # --- Correct Category Handling ---
    if dataset_name == 'Personal_Finance_Dataset1.csv':
        if df.shape[1] >= 6:
            df['Category'] = df.iloc[:, 5].astype(str).str.strip().str.title()  # Column 6 = Sub Category
        else:
            df['Category'] = 'Unknown'
    elif dataset_name == 'Personal_Finance_Dataset2.csv':
        if 'Category' in df.columns:
            df['Category'] = df['Category'].astype(str).str.strip().str.title()
        else:
            df['Category'] = 'Unknown'

    # Rename 'Type' to 'Category Type' for Dataset2
    if dataset_name == 'Personal_Finance_Dataset2.csv' and 'Type' in df.columns:
        df.rename(columns={'Type': 'Category Type'}, inplace=True)

    # Clean Category Type
    if 'Category Type' in df.columns:
        df['Category Type'] = df['Category Type'].astype(str).str.strip().str.title()
    else:
        df['Category Type'] = 'Unknown'

    return df

# --- MAIN EXECUTION ---
for raw_file, output_file in files:
    print(f"\nProcessing: {raw_file}")
    raw_path = os.path.join(RAW_DIR, raw_file)
    output_path = os.path.join(PROCESSED_DIR, output_file)

    df = pd.read_csv(raw_path)
    print("Original shape:", df.shape)

    cleaned_df = CleaningFunction(df, raw_file)
    print("Cleaned shape:", cleaned_df.shape)

    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")