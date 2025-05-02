import os
import pandas as pd

# Benchmark evaluation logic
def summarize_thresholds(df, model_name, output_base='benchmarks'):
    """
    Evaluates prediction benchmarks (50%, 100%, 150%) across original and synthetic users.
    Saves threshold_summary.csv in benchmarks/{model_name}/
    """
    thresholds = [col for col in df.columns if col.startswith("Threshold_")]
    segments = {
        'All': df,
        'Original': df[df['Source'] == 'original'] if 'Source' in df.columns else df,
        'Synthetic': df[df['Source'] == 'synthetic'] if 'Source' in df.columns else df
    }

    records = []
    for segment_name, subset in segments.items():
        for threshold in thresholds:
            if threshold in subset:
                try:
                    rate = subset[threshold].mean() * 100
                    records.append({
                        'Model': model_name,
                        'Segment': segment_name,
                        'Threshold': threshold.replace('Threshold_', '') + '%',
                        'Percent_Met': round(rate, 2)
                    })
                except Exception as e:
                    print(f"Warning: Failed to process {threshold} for {segment_name}: {e}")

    summary_df = pd.DataFrame(records)
    output_path = os.path.join(output_base, model_name)
    os.makedirs(output_path, exist_ok=True)
    summary_df.to_csv(os.path.join(output_path, 'threshold_summary.csv'), index=False)
    print(f"Saved benchmark summary to {output_path}/threshold_summary.csv")
    return summary_df


# Auto-run for all known models if this file is executed directly
if __name__ == "__main__":
    MODEL_NAMES = [
        "svr", "random_forest", "linear_regression",
        "svr_blindtest", "random_forest_blindtest", "linear_regression_blindtest"
    ]

    MODEL_PATHS = {}
    for model in MODEL_NAMES:
        segmented = f"outputs/{model}/predictions_segmented.csv"
        fallback = f"outputs/{model}/predictions.csv"
        MODEL_PATHS[model] = segmented if os.path.exists(segmented) else fallback

    for model_name, csv_path in MODEL_PATHS.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Rename prediction column if needed
            if "Predicted" in df.columns and "Predicted_Net_Savings" not in df.columns:
                df = df.rename(columns={"Predicted": "Predicted_Net_Savings"})

            # Rename actual column if needed
            actual_aliases = ["Actual", "Savings_Rate", "Actual_Net_Savings", "Net_Savings"]
            found_actual = next((col for col in actual_aliases if col in df.columns), None)

            if found_actual and found_actual != "Actual":
                df = df.rename(columns={found_actual: "Actual"})
            elif not found_actual:
                raise KeyError("No valid column found for actual values (e.g., 'Actual', 'Savings_Rate', 'Actual_Net_Savings', 'Net_Savings').")

            # Add threshold flags if not already there
            if not any(col.startswith("Threshold_") for col in df.columns):
                for t in [0.5, 1.0, 1.5]:
                    label = f"Threshold_{int(t * 100)}"
                    df[label] = df["Predicted_Net_Savings"] >= (df["Actual"] * t)

            summarize_thresholds(df, model_name=model_name)
        else:
            print(f"Missing: {csv_path}")
