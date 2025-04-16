import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the summary data
summary_path = "outputs/model_comparison_summary.csv"
summary_df = pd.read_csv(summary_path)

# Setup plot styling
sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# Function to annotate bars
def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black')

# Filter options
def plot_metrics(segment='All Data'):
    df_filtered = summary_df[summary_df['Segment'] == segment]
    metrics = ['MAE', 'RMSE', 'R2']

    # Create output dir for segment
    segment_clean = segment.replace(" ", "_").lower()
    output_dir = os.path.join("outputs", f"charts_{segment_clean}")
    os.makedirs(output_dir, exist_ok=True)

    # Individual plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_filtered, x="Model", y=metric, hue="Dataset")
        annotate_bars(ax)
        plt.title(f"{metric} by Model and Dataset ({segment})")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_{metric.lower()}.png"))
        plt.close()

    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(metrics):
        ax = sns.barplot(data=df_filtered, x="Model", y=metric, hue="Dataset", ax=axes[i])
        annotate_bars(ax)
        axes[i].set_title(metric)
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_model_comparison.png"))
    plt.close()

# Generate charts for each data segment
for segment in ['All Data', 'Original Only', 'Synthetic Only']:
    plot_metrics(segment)

print("Charts generated in 'outputs/charts_all_data', 'charts_original_only', and 'charts_synthetic_only'")
