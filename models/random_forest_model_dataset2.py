import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN
from scripts.benchmark_summary import summarize_thresholds

sns.set_theme(style="whitegrid")

# Load and sort data
df = load_data(TABLE_TRAIN).sort_values('Date')

# Time-based split
X = df[FEATURES]
y = df[TARGET]
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"First train date: {df.iloc[0]['Date']}, Last train date: {df.iloc[split_index - 1]['Date']}")
print(f"First test date: {df.iloc[split_index]['Date']}, Last test date: {df.iloc[-1]['Date']}")

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
model_dir = 'models/random_forest'
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, 'model.pkl'))

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Random Forest Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save outputs
output_dir = 'outputs/random_forest'
os.makedirs(output_dir, exist_ok=True)

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Net Savings')
plt.ylabel('Predicted Net Savings')
plt.title('Actual vs Predicted Net Savings (Random Forest - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.show()

# Save predictions
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Add benchmark threshold flags
results['Threshold_50'] = results['Predicted'] >= (results['Actual'] * 0.5)
results['Threshold_100'] = results['Predicted'] >= results['Actual']
results['Threshold_150'] = results['Predicted'] >= (results['Actual'] * 1.5)
results['Source'] = 'original'  # All training data is assumed to be original

# Save benchmark summary
summarize_thresholds(results, model_name='random_forest')

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- All Data ---\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.2f}\n")
    f.write("\n--- Original Only ---\n")
    f.write("MAE: N/A\n")
    f.write("RMSE: N/A\n")
    f.write("R2: N/A\n")
    f.write("\n--- Synthetic Only ---\n")
    f.write("MAE: N/A\n")
    f.write("RMSE: N/A\n")
    f.write("R2: N/A\n")

print(f"Random Forest model training complete. Results saved to {output_dir}/")
