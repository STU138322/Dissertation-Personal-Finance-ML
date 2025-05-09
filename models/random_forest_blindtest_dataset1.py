import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_BLINDTEST
from scripts.benchmark_summary import summarize_thresholds

sns.set_theme(style="whitegrid")

# === Load model ===
model = load('models/random_forest/model.pkl')

# === Load Dataset1 blind test data ===
df = load_data(TABLE_BLINDTEST).sort_values("Date")
X_test = df[FEATURES]
y_test = df[TARGET]

# === Predict ===
df['Predicted'] = model.predict(X_test)

# === Remove NaNs in prediction or target ===
df = df.dropna(subset=[TARGET, 'Predicted'])
y_test = df[TARGET]

# === Segment: Real vs Synthetic ===
real = df[df['Source'] == 'original']
synthetic = df[df['Source'] == 'synthetic']

# === Evaluation ===
def evaluate(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R2': round(r2_score(y_true, y_pred), 4)
    }

metrics_all = evaluate(y_test, df['Predicted'])
metrics_real = evaluate(real[TARGET], real['Predicted']) if len(real) > 0 else {}
metrics_synth = evaluate(synthetic[TARGET], synthetic['Predicted']) if len(synthetic) > 0 else {}

# === Save outputs ===
output_dir = 'outputs/random_forest_blindtest'
os.makedirs(output_dir, exist_ok=True)

# Save metrics
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    f.write("--- All Data ---\n")
    for k, v in metrics_all.items():
        f.write(f"{k}: {v}\n")
    if metrics_real:
        f.write("\n--- Original Only ---\n")
        for k, v in metrics_real.items():
            f.write(f"{k}: {v}\n")
    if metrics_synth:
        f.write("\n--- Synthetic Only ---\n")
        for k, v in metrics_synth.items():
            f.write(f"{k}: {v}\n")

# Save predictions
df[['Date', 'Source', TARGET, 'Predicted']].to_csv(
    os.path.join(output_dir, 'predictions_segmented.csv'), index=False
)

# Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df[TARGET], y=df['Predicted'], hue=df['Source'])
plt.xlabel("Actual Savings Rate")
plt.ylabel("Predicted Savings Rate")
plt.title("Random Forest - Blind Test (Dataset1)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_colored.png'))
plt.close()

# Benchmarking
df['Threshold_50'] = df['Predicted'] >= (df[TARGET] * 0.5)
df['Threshold_100'] = df['Predicted'] >= df[TARGET]
df['Threshold_150'] = df['Predicted'] >= (df[TARGET] * 1.5)
summarize_thresholds(df, model_name='random_forest')

print(f"Random Forest blind test complete. Results saved to {output_dir}/")
