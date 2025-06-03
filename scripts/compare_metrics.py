import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'outputs/'
records = []

segment_alias = {
    "Test Set Performance": "All Data",
    "All Data": "All Data",
    "Original Only": "Original Only",
    "Synthetic Only": "Synthetic Only"
}

# Parse all metrics.txt files
for folder in os.listdir(base_dir):
    subdir = os.path.join(base_dir, folder)
    metrics_file = os.path.join(subdir, "metrics.txt")
    if os.path.isfile(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                lines = f.readlines()

            model = folder.replace("_blindtest", "").replace("_", " ").title().replace("Dataset1", "").strip()
            dataset = "Dataset1" if "blindtest" in folder.lower() else "Dataset2"

            segment = None
            for line in lines:
                line = line.strip()
                if "---" in line:
                    raw_segment = line.replace("-", "").strip()
                    segment = segment_alias.get(raw_segment, raw_segment)
                elif ":" in line and segment:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if val not in ["N/A", "None", ""]:
                        try:
                            val = float(val)
                            records.append({
                                "Model": model,
                                "Dataset": dataset,
                                "Segment": segment,
                                "Metric": key,
                                "Value": val
                            })
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Failed to read {metrics_file}: {e}")

# Create and save enhanced comparison summary
if records:
    df = pd.DataFrame(records)
    summary = df.pivot_table(
        index=["Model", "Dataset", "Segment"],
        columns="Metric",
        values="Value"
    ).reset_index()

    summary = summary.round(4)
    summary = summary.sort_values(by=["Dataset", "Model", "Segment"])

    # Identify best performers
    best_summary = summary[summary["Segment"] == "All Data"].copy()

    best_mae_idx = best_summary["MAE"].idxmin()
    best_rmse_idx = best_summary["RMSE"].idxmin()
    best_r2_idx = best_summary["R2"].idxmax()

    summary["Best MAE"] = summary.index == best_mae_idx
    summary["Best RMSE"] = summary.index == best_rmse_idx
    summary["Best R2"] = summary.index == best_r2_idx

    # Save to CSV
    summary_path = os.path.join(base_dir, "model_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Save best results to TXT
    with open(os.path.join(base_dir, "model_comparison_summary.txt"), "w") as f:
        f.write("=== Best Performing Models (All Data Segment) ===\n")
        f.write(f"Best MAE  : {summary.loc[best_mae_idx, 'Model']} on {summary.loc[best_mae_idx, 'Dataset']} (MAE: {summary.loc[best_mae_idx, 'MAE']})\n")
        f.write(f"Best RMSE : {summary.loc[best_rmse_idx, 'Model']} on {summary.loc[best_rmse_idx, 'Dataset']} (RMSE: {summary.loc[best_rmse_idx, 'RMSE']})\n")
        f.write(f"Best R²   : {summary.loc[best_r2_idx, 'Model']} on {summary.loc[best_r2_idx, 'Dataset']} (R²: {summary.loc[best_r2_idx, 'R2']})\n")

    # Plot bar chart for visual comparison
    chart_data = best_summary[["Model", "Dataset", "MAE", "RMSE", "R2"]]
    chart_data.set_index(["Model", "Dataset"]).plot(kind='bar', figsize=(10, 6))
    plt.title("Model Performance (All Data Segment)")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "model_comparison_chart.png"))

    #Generate separate subplot figure per dataset
    for dataset_name in summary["Dataset"].unique():
        data_subset = best_summary[best_summary["Dataset"] == dataset_name]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for idx, metric in enumerate(["MAE", "RMSE", "R2"]):
            axs[idx].bar(data_subset["Model"], data_subset[metric], color='steelblue')
            axs[idx].set_title(metric)
            if metric == "R2":
                axs[idx].set_ylim(-3.5, 1)
            else:
                max_val = data_subset[metric].max()
                axs[idx].set_ylim(0, max_val * 1.1)
            axs[idx].set_ylabel(metric)
            axs[idx].set_xlabel("Model")

        fig.suptitle(f"Model Performance Comparison ({dataset_name} - All Data Segment)", fontsize=14)
        plt.tight_layout()
        subplot_path = os.path.join(base_dir, f"model_comparison_subplots_{dataset_name.lower()}.png")
        plt.savefig(subplot_path)
        plt.close()

        print(f" - Subplot saved: {subplot_path}")

    print("Summary saved to:")
    print(" - model_comparison_summary.csv")
    print(" - model_comparison_summary.txt")
    print(" - model_comparison_chart.png")

else:
    print("No metric records found.")
