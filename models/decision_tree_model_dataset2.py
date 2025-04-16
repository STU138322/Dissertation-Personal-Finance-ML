import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# Load and sort data
df = load_data(TABLE_TRAIN).sort_values('Date')

X = df[FEATURES]
y = df[TARGET]
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"First train date: {df.iloc[0]['Date']}, Last train date: {df.iloc[split_index - 1]['Date']}")
print(f"First test date: {df.iloc[split_index]['Date']}, Last test date: {df.iloc[-1]['Date']}")

# Initialize and train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
model_dir = 'models/decision_tree'
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, 'model.pkl'))

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Decision Tree Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save outputs
output_dir = 'outputs/decision_tree'
os.makedirs(output_dir, exist_ok=True)

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Net Savings')
plt.ylabel('Predicted Net Savings')
plt.title('Actual vs Predicted Net Savings (Decision Tree - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.show()

# Save predictions
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

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

print(f"Decision Tree model training complete. Results saved to {output_dir}/")
