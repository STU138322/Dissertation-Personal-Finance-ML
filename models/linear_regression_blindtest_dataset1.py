import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import sys

# Load project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_BLINDTEST

sns.set_theme(style="whitegrid")

# === Load model ===
model = load('models/linear_regression/model.pkl')

# === Load blind test data (Dataset1) ===
df = load_data(TABLE_BLINDTEST).sort_values("Date")
X_test = df[FEATURES]
y_test = df[TARGET]

# === Predict ===
df['Predicted'] = model.predict(X_test)

# === Remove NaN targets (e.g., from Savings_Rate = Net_Savings / 0 Income) ===
df = df.dropna(subset=[TARGET, 'Predicted'])
y_test = df[TARGET]

# === Segment: Real vs Synthetic ===
real = df[df['Source'] == 'original']
synthetic = df[df['Source'] == 'synthetic']

# === Evaluation function ===
def evaluate(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R2': round(r2_score(y_true, y_pred), 4)
    }

# === Evaluate results ===
metrics_all = evaluate(y_test, df['Predicted'])
metrics_real = evaluate(real[TARGET], real['Predicted']) if len(real) > 0 else {}
metrics_synth = evaluate(synthetic[TARGET], synthetic['Predicted']) if len(synthetic) > 0 else {}

# === Save outputs ===
output_dir = 'outputs/linear_regression_blindtest'
os.makedirs(output_dir, exist_ok=True)

# Save segmented metrics
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

# Save standardized prediction output
df["Actual"] = df[TARGET]
df_segment = df[["Date", "Source", "Actual", "Predicted"]]
df_segment.to_csv(os.path.join(output_dir, 'predictions_segmented.csv'), index=False)

# Save colored scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Actual"], y=df["Predicted"], hue=df['Source'])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression - Blind Test (Dataset1)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_colored.png'))
plt.close()

print(f"Linear Regression blind test complete. Results saved to {output_dir}/")
