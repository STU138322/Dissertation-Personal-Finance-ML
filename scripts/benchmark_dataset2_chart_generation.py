import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_chart_generation():
    # Settings
    benchmark_dir = "benchmarks"
    model_folders = [
        "linear_regression",
        "random_forest",
        "svr"
    ]

    # Load and combine benchmark summaries
    all_dfs = []
    for model in model_folders:
        path = os.path.join(benchmark_dir, model, "threshold_summary.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Model"] = model.replace("_", " ").title()
            all_dfs.append(df)
        else:
            print(f"Warning: Missing benchmark file for {model}")

    if not all_dfs:
        raise FileNotFoundError("No benchmark summary CSVs found in expected folders.")

    combined_df = pd.concat(all_dfs).reset_index(drop=True)

    # Clean and organize
    combined_df["Threshold"] = combined_df["Threshold"].str.replace("%", "")
    combined_df["Percent_Met"] = combined_df["Percent_Met"].astype(float)
    combined_df["Segment"] = combined_df["Segment"].astype(str)
    combined_df["Threshold"] = combined_df["Threshold"].astype(str)

    # Plotting
    sns.set(style="whitegrid")
    g = sns.catplot(
        data=combined_df,
        kind="bar",
        x="Model",
        y="Percent_Met",
        hue="Threshold",
        col="Segment",
        height=5,
        aspect=1,
        palette="muted"
    )

    g.set_axis_labels("Model", "Percent of Records Meeting Threshold")
    g.set_titles("{col_name} Segment")
    g.set(ylim=(0, 110))

    # Annotate bars
    for ax in g.axes.flatten():
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

    plt.subplots_adjust(top=0.85)
    g.figure.suptitle("Benchmark Evaluation: Savings Threshold Accuracy by Model", fontsize=16)

    # Save output
    output_path = os.path.join(benchmark_dir, "plot_dataset2_threshold_summary.png")
    g.savefig(output_path)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    run_chart_generation()
