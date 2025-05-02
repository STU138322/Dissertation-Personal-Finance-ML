import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import sys
import argparse

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

# === Growth Tracking Function Dataset2 ===
def run_growth_tracking(model_name, model_path):
    print(f"\nRunning growth tracker for: {model_name}")

    model = load(model_path)
    df = load_data(TABLE_TRAIN).sort_values("Date")
    df["Predicted_Savings_Rate"] = model.predict(df[FEATURES])

    thresholds = [0.5, 1.0, 1.5]
    for t in thresholds:
        label = f"Threshold_{int(t * 100)}"
        df[label] = df["Predicted_Savings_Rate"] >= (df[TARGET] * t)

    output_dir = f"predicted_savings_tracking/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_cols = ["Date", "Category", "Income", "Expense", TARGET, "Predicted_Savings_Rate"]
    output_cols += [f"Threshold_{int(t * 100)}" for t in thresholds]

    df[output_cols].to_csv(os.path.join(output_dir, "user_growth_tracking.csv"), index=False)

    # Plot growth over time
    df_sorted = df.sort_values("Date").copy()
    df_sorted["Actual_Savings_Rate"] = df_sorted[TARGET].rolling(window=7, min_periods=1).mean()
    df_sorted["Predicted_Savings_Rate_Smoothed"] = df_sorted["Predicted_Savings_Rate"].rolling(window=7, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted["Date"], df_sorted["Actual_Savings_Rate"], label="Smoothed Actual", color="blue", linestyle="-")
    plt.plot(df_sorted["Date"], df_sorted["Predicted_Savings_Rate_Smoothed"], label="Smoothed Predicted", color="orange", linestyle="--")

    tick_interval = max(len(df_sorted) // 10, 1)
    xticks = df_sorted["Date"].iloc[::tick_interval]
    xtick_labels = pd.to_datetime(xticks).dt.strftime('%Y-%m-%d')
    plt.xticks(ticks=xticks, labels=xtick_labels, rotation=45)

    plt.title(f"Smoothed Savings Rate Over Time – {model_name.replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel("Savings Rate (Smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "growth_over_time.png"))
    plt.close()

    print(f"{model_name} growth tracking complete → {output_dir}/")


# === Growth Tracking Function Dataset1 ===
def run_blindtest_growth_tracking():
    print("\nRunning blindtest growth tracking for Dataset1...")
    dataset1_path = "data/engineered/engineered_dataset1.csv"
    df_full = pd.read_csv(dataset1_path, parse_dates=["Date"]).dropna().sort_values("Date")

    model_configs = {
        "linear_regression_blindtest": "models/linear_regression/model.pkl",
        "random_forest_blindtest": "models/random_forest/model.pkl",
        "svr_blindtest": "models/svr/model.pkl"
    }

    for name, model_path in model_configs.items():
        print(f"  - Running: {name}")
        model = load(model_path)
        df = df_full.copy()
        df["Predicted_Savings_Rate"] = model.predict(df[FEATURES])

        df_sorted = df.sort_values("Date").copy()
        df_sorted["Actual_Savings_Rate"] = df_sorted[TARGET].rolling(window=7, min_periods=1).mean()
        df_sorted["Predicted_Savings_Rate_Smoothed"] = df_sorted["Predicted_Savings_Rate"].rolling(window=7, min_periods=1).mean()

        output_dir = f"predicted_savings_tracking_blindtest/{name}"
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(df_sorted["Date"], df_sorted["Actual_Savings_Rate"], label="Smoothed Actual", color="green", linestyle="-")
        plt.plot(df_sorted["Date"], df_sorted["Predicted_Savings_Rate_Smoothed"], label="Smoothed Predicted", color="red", linestyle="--")

        tick_interval = max(len(df_sorted) // 10, 1)
        xticks = df_sorted["Date"].iloc[::tick_interval]
        xtick_labels = pd.to_datetime(xticks).dt.strftime('%Y-%m-%d')
        plt.xticks(ticks=xticks, labels=xtick_labels, rotation=45)

        plt.title(f"Smoothed Savings Rate Over Time – {name.replace('_', ' ').title()}")
        plt.xlabel("Date")
        plt.ylabel("Savings Rate (Smoothed)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "growth_over_time.png"))
        plt.close()

        df[["Date", "Category", TARGET, "Predicted_Savings_Rate"]].to_csv(
            os.path.join(output_dir, "user_growth_tracking.csv"), index=False
        )

        print(f"{name} complete → {output_dir}/")


# === Optional CLI Usage ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run growth tracker for all models (Dataset2)")
    parser.add_argument("--blindtest", action="store_true", help="Run blindtest growth tracking for Dataset1")
    args = parser.parse_args()

    if args.all:
        for model, path in MODELS.items():
            run_growth_tracking(model, path)
    if args.blindtest:
        run_blindtest_growth_tracking()
    if not args.all and not args.blindtest:
        print("Use --all and/or --blindtest to run growth tracking.")
