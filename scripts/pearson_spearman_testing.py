import os
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, TABLE_TRAIN, TABLE_BLINDTEST

# Define feature pairs for correlation testing
feature_pairs = [
    ('Income', 'Savings_Rate'),
    ('Expense', 'Savings_Rate'),
    ('Rolling_Income', 'Savings_Rate'),
    ('Rolling_Expense', 'Savings_Rate'),
    ('Income', 'Expense')
]

# Output directory
OUTPUT_DIR = 'notebooks/hypothesis_tests'
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, 'correlation_results.txt')

# === Test function ===
def run_tests(df, dataset_name):
    result = f"\n=== Correlation Tests for {dataset_name} ===\n"
    for x, y in feature_pairs:
        result += f"\nTesting {x} vs {y}:\n"

        if x not in df.columns or y not in df.columns:
            result += f"⚠️ Skipped: one or both columns missing.\n"
            continue

        subset = df[[x, y]].dropna()

        if len(subset) < 10:
            result += f"⚠️ Skipped: not enough valid records.\n"
            continue

        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(subset[x], subset[y])
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(subset[x], subset[y])

        result += f"Pearson Correlation: r = {pearson_corr:.4f}, p-value = {pearson_p:.4g}\n"
        result += f"Spearman Correlation: r = {spearman_corr:.4f}, p-value = {spearman_p:.4g}\n"
    return result

# Load datasets
train_data = load_data(TABLE_TRAIN)
blindtest_data = load_data(TABLE_BLINDTEST)

# Run and save results
all_results = ""
all_results += run_tests(train_data, 'Dataset2 (Training Set)')
all_results += run_tests(blindtest_data, 'Dataset1 (Blindtest Set)')

with open(output_file, 'w') as f:
    f.write(all_results)

print(f"\nCorrelation test results saved to: {output_file}")
