import os
import pandas as pd

base_dir = 'outputs/'  # relative to project root
records = []

# Include additional model folders
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
                    segment = line.replace("-", "").strip()
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

# Create summary DataFrame
if records:
    df = pd.DataFrame(records)
    summary = df.pivot_table(
        index=["Model", "Dataset", "Segment"],
        columns="Metric",
        values="Value"
    ).reset_index()

    # Save to CSV
    summary_path = os.path.abspath(os.path.join(base_dir, "model_comparison_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print("Summary saved to model_comparison_summary.csv")
else:
    print("No metric records found to summarize.")