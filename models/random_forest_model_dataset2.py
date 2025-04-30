import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN
from scripts.benchmark_summary import summarize_thresholds

sns.set_theme(style="whitegrid")

# === Load and prepare data ===
df = load_data(TABLE_TRAIN).dropna().sort_values("Date")
X = df[FEATURES]
y = df[TARGET]  # Now using 'Savings_Rate'

# === Time-based split (80/20) ===
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# === Train model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save model ===
model_dir = 'models/random_forest'
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, 'model.pkl'))

# === Predict ===
y_pred = model.predict(X_test)

# === Evaluate ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Random Forest Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === Cross-validation ===
def rmse_score(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    'MAE': make_scorer(mean_absolute_error),
    'RMSE': make_scorer(rmse_score),
    'R2': make_scorer(r2_score)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, scorer in scorers.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
    cv_results[name] = {
        'folds': [round(s, 2) for s in scores],
        'mean': round(np.mean(scores), 2),
        'std': round(np.std(scores), 2)
    }

# === Save Outputs ===
output_dir = 'outputs/random_forest'
os.makedirs(output_dir, exist_ok=True)

# Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Savings Rate')
plt.ylabel('Predicted Savings Rate')
plt.title('Actual vs Predicted Savings Rate (Random Forest - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

# Feature importance
importances = model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=FEATURES)
plt.title("Feature Influence on Savings Rate Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Benchmarking
results['Threshold_50'] = results['Predicted'] >= (results['Actual'] * 0.5)
results['Threshold_100'] = results['Predicted'] >= results['Actual']
results['Threshold_150'] = results['Predicted'] >= (results['Actual'] * 1.5)
results['Source'] = 'original'
summarize_thresholds(results, model_name='random_forest')

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- Test Set Performance ---\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.2f}\n")

# Save CV metrics
with open(os.path.join(output_dir, "cv_metrics.txt"), "w") as f:
    f.write("=== 5-Fold Cross-Validation Results (Random Forest) ===\n")
    for metric, stats in cv_results.items():
        f.write(f"\n{metric}:\n")
        f.write("  Individual Scores: " + ", ".join(map(str, stats['folds'])) + "\n")
        f.write(f"  Mean = {stats['mean']} | Std = {stats['std']}\n")

print(f"Random Forest training complete. Outputs saved to {output_dir}/")
