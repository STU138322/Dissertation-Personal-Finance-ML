import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import sys

# --- Project Config ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# === Available Models ===
MODELS = {
    "linear_regression": "models/linear_regression/model.pkl",
    "random_forest": "models/random_forest/model.pkl",
    "svr": "models/svr/model.pkl"
}

# === Growth Tracking Function ===
def run_growth_tracking(model_name, model_path):
    print(f"\nRunning growth tracker for: {model_name}")

    # Load trained model
    model = load(model_path)

    # Load Dataset2
    df = load_data(TABLE_TRAIN).sort_values("Date")
    df["Predicted_Net_Savings"] = model.predict(df[FEATURES])

    # Add Threshold Flags
    df["Threshold_50"] = df["Predicted_Net_Savings"] >= (df[TARGET] * 0.5)
    df["Threshold_100"] = df["Predicted_Net_Savings"] >= df[TARGET]
    df["Threshold_150"] = df["Predicted_Net_Savings"] >= (df[TARGET] * 1.5)

    # Output Directory
    output_dir = f"outputs/{model_name}_growth_tracker"
    os.makedirs(output_dir, exist_ok=True)

    # Save user growth tracking CSV (including Category)
    df[["Date", "Category", "Income", "Expense", TARGET, "Predicted_Net_Savings",
        "Threshold_50", "Threshold_100", "Threshold_150"]].to_csv(
        os.path.join(output_dir, "user_growth_tracking.csv"), index=False
    )

    # Plot growth over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df[TARGET], label="Actual Net Savings", marker="o")
    plt.plot(df["Date"], df["Predicted_Net_Savings"], label="Predicted Net Savings", marker="x")
    plt.title(f"Net Savings Over Time – {model_name.replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel("Net Savings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "growth_over_time.png"))
    plt.close()

    print(f"{model_name} growth tracking complete → {output_dir}/")
