import os
import pandas as pd

# Benchmark evaluation logic
def summarize_thresholds(df, model_name, output_base='benchmarks'):
    """
    Evaluates prediction benchmarks (50%, 100%, 150%) across original and synthetic users.

    Parameters:
    - df: DataFrame with actual, predicted, threshold flags, and Source column
    - model_name: string for naming the output folder
    - output_base: base path for output storage

    Saves: threshold_summary.csv in a folder under output_base/model_name
    """
    thresholds = ['Threshold_50', 'Threshold_100', 'Threshold_150']
    segments = {
        'All': df,
        'Original': df[df['Source'] == 'original'],
        'Synthetic': df[df['Source'] == 'synthetic']
    }

    records = []
    for segment_name, subset in segments.items():
        for threshold in thresholds:
            if threshold in subset:
                rate = subset[threshold].mean() * 100
                records.append({
                    'Model': model_name,
                    'Segment': segment_name,
                    'Threshold': threshold.replace('Threshold_', '') + '%',
                    'Percent_Met': round(rate, 2)
                })

    summary_df = pd.DataFrame(records)
    output_path = os.path.join(output_base, model_name)
    os.makedirs(output_path, exist_ok=True)
    summary_df.to_csv(os.path.join(output_path, 'threshold_summary.csv'), index=False)
    print(f"Saved benchmark summary to {output_path}/threshold_summary.csv")
    return summary_df