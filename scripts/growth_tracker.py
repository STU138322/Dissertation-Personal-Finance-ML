import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# --- Project Config ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# === Available Models ===
MODELS = {
    "linear_regression": "models/linear_regression/model.pkl",
    "decision_tree": "models/decision_tree/model.pkl",
    "random_forest": "models/random_forest/model.pkl"
}

def run_growth_tracking(model_name, model_path):
    print(f"\n🔍 Running growth tracker for: {model_name}")
    
    # Load model
    model = load(model_path)

    # Load dataset2 (1-user time series)
    df = load_data(TABLE_TRAIN).sort_values("Date")
    df["Predicted_Net_Savings"] = model.predict(df[FEATURES])

    # Threshold evaluations
    df["Threshold_50"] = df["Predicted_Net_Savings"] >= (df[TARGET] * 0.5)
    df["Threshold_100"] = df["Predicted_Net_Savings"] >= df[TARGET]
    df["Threshold_150"] = df["Predicted_Net_Savings"] >= (df[TARGET] * 1.5)

    # Output directory
    output_dir = f"outputs/{model_name}_growth_tracker"
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    df[["Date", "Income", "Expense", TARGET, "Predicted_Net_Savings",
        "Threshold_50", "Threshold_100", "Threshold_150"]].to_csv(
        os.path.join(output_dir, "user_growth_tracking.csv"), index=False
    )

    # Save plot
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

    print(f"{model_name} tracking complete → saved to {output_dir}/")

# === CLI Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run growth tracker for a specific model or all models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, choices=MODELS.keys(), help="Model to evaluate")
    group.add_argument("--all", action="store_true", help="Run all models")

    args = parser.parse_args()

    if args.all:
        for model, path in MODELS.items():
            run_growth_tracking(model, path)
    else:
        run_growth_tracking(args.model, MODELS[args.model])
