import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load trained model
model = load('models/random_forest/model.pkl')

# Load test data
df = pd.read_csv('data/engineered/engineered_dataset1.csv', parse_dates=['Date']).dropna()

# Define features
features = ['Income', 'Expense', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']
X_test = df[features]
y_test = df['Net_Savings']

# Predict
df['Predicted'] = model.predict(X_test)
df['Threshold_50'] = df['Predicted'] >= (df['Net_Savings'] * 0.5)
df['Threshold_100'] = df['Predicted'] >= df['Net_Savings']
df['Threshold_150'] = df['Predicted'] >= (df['Net_Savings'] * 1.5)

# Segment by Source
real = df[df['Source'] == 'original']
synthetic = df[df['Source'] == 'synthetic']

def evaluate(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2': round(r2_score(y_true, y_pred), 2)
    }

metrics_all = evaluate(df['Net_Savings'], df['Predicted'])
metrics_real = evaluate(real['Net_Savings'], real['Predicted'])
metrics_synth = evaluate(synthetic['Net_Savings'], synthetic['Predicted'])

output_dir = 'outputs/random_forest_blindtest'
os.makedirs(output_dir, exist_ok=True)

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

df.to_csv(os.path.join(output_dir, 'predictions_segmented.csv'), index=False)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Net_Savings'], y=df['Predicted'], hue=df['Source'])
plt.xlabel('Actual Net Savings')
plt.ylabel('Predicted Net Savings')
plt.title('Random Forest Blind Test - Dataset1')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_colored.png'))
plt.close()

print(f"Random Forest blind test complete. Results saved to: {output_dir}/")
