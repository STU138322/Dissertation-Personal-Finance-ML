import pandas as pd
import os

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Mapping files
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

    # Set Category properly
    if dataset_name == 'Personal_Finance_Dataset1.csv':
        if df.shape[1] >= 7:
            df['Category'] = df.iloc[:, 6].astype(str).str.strip().str.title()
        else:
            print(f"[Warning] Column 7 not found in {dataset_name}. Setting Category to 'Unknown'.")
            df['Category'] = 'Unknown'
    else:
        # For Dataset2, just clean existing Category
        if 'Category' in df.columns:
            df['Category'] = df['Category'].astype(str).str.strip().str.title()
        else:
            print(f"[Warning] 'Category' column not found in {dataset_name}. Setting to 'Unknown'.")
            df['Category'] = 'Unknown'

    # Special handling for Dataset2 to rename 'Type' → 'Category Type'
    if dataset_name == 'Personal_Finance_Dataset2.csv':
        if 'Type' in df.columns:
            df.rename(columns={'Type': 'Category Type'}, inplace=True)

    # Clean Category Type
    if 'Category Type' in df.columns:
        df['Category Type'] = df['Category Type'].astype(str).str.strip().str.title()
    else:
        df['Category Type'] = 'Unknown'

    return df


# Run cleaning
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
