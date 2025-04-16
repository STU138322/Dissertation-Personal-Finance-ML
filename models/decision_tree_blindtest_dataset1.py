import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_connect import load_data, FEATURES, TARGET, TABLE_BLINDTEST

sns.set_theme(style="whitegrid")

# Load trained model
model = load('models/decision_tree/model.pkl')

# Load blindtest data
df = load_data(TABLE_BLINDTEST).sort_values('Date')
X_test = df[FEATURES]
y_test = df[TARGET]

# Predict
y_pred = model.predict(X_test)

# Add predictions and thresholds
df['Predicted'] = y_pred
df['Threshold_50'] = df['Predicted'] >= (df['Net_Savings'] * 0.5)
df['Threshold_100'] = df['Predicted'] >= df['Net_Savings']
df['Threshold_150'] = df['Predicted'] >= (df['Net_Savings'] * 1.5)

# Segment by Source
real = df[df['Source'] == 'original']
synthetic = df[df['Source'] == 'synthetic']

# Evaluation function
def evaluate(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2': round(r2_score(y_true, y_pred), 2)
    }

# Compute metrics
metrics_all = evaluate(df['Net_Savings'], df['Predicted'])
metrics_real = evaluate(real['Net_Savings'], real['Predicted'])
metrics_synth = evaluate(synthetic['Net_Savings'], synthetic['Predicted'])

# Output dir
output_dir = 'outputs/decision_tree_blindtest'
os.makedirs(output_dir, exist_ok=True)

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- All Data ---\n")
    for k, v in metrics_all.items():
        f.write(f"{k}: {v}\n")
    f.write("\n--- Original Only ---\n")
    for k, v in metrics_real.items():
        f.write(f"{k}: {v}\n")
    f.write("\n--- Synthetic Only ---\n")
    for k, v in metrics_synth.items():
        f.write(f"{k}: {v}\n")

# Save predictions
df.to_csv(os.path.join(output_dir, 'predictions_segmented.csv'), index=False)

# Plot full actual vs predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Net_Savings'], y=df['Predicted'], hue=df['Source'])
plt.xlabel('Actual Net Savings')
plt.ylabel('Predicted Net Savings')
plt.title('Decision Tree Blind Test - Dataset1')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_colored.png'))
plt.close()

print(f"Blind test complete. Results saved to: {output_dir}/")
