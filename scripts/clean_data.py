import pandas as pd
import os

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

files = [
    ('Personal_Finance_Dataset1.csv', 'cleaned_dataset1.csv'),
    ('Personal_Finance_Dataset2.csv', 'cleaned_dataset2.csv')
]

def CleaningFunction(df):
    
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=['Date', 'Amount'])

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount'])

    if 'Category' in df.columns:
        df['Category'] = df['Category'].str.strip().str.title()

    return df

for raw_file, output_file in files:
    print(f"\nProcessing: {raw_file}")
    raw_path = os.path.join(RAW_DIR, raw_file)
    output_path = os.path.join(PROCESSED_DIR, output_file)

    df = pd.read_csv(raw_path)
    print("Original shape:", df.shape)

    cleaned_df = CleaningFunction(df)
    print("Cleaned shape:", cleaned_df.shape)

    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
