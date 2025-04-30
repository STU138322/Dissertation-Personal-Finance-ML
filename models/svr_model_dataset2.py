import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# === Load Data ===
df = load_data(TABLE_TRAIN).dropna().sort_values('Date')
X = df[FEATURES]
y = df[TARGET]  # Now 'Savings_Rate'

# === 70:30 Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train Model ===
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train, y_train)

# === Save Model ===
model_dir = 'models/svr'
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, 'model.pkl'))

# === Predict ===
y_pred = model.predict(X_test)

# === Evaluate on Test Set ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- SVR Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# === Cross-Validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_score(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    'MAE': make_scorer(mean_absolute_error),
    'RMSE': make_scorer(rmse_score),
    'R2': make_scorer(r2_score)
}

cv_results = {}
for name, scorer in scorers.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
    rounded_scores = [round(s, 2) for s in scores]
    cv_results[name] = {
        'folds': rounded_scores,
        'mean': round(np.mean(scores), 2),
        'std': round(np.std(scores), 2)
    }

# === Save Outputs ===
output_dir = 'outputs/svr'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Savings Rate')
plt.ylabel('Predicted Savings Rate')
plt.title('Actual vs Predicted Savings Rate (SVR - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv(
    os.path.join(output_dir, 'predictions.csv'), index=False
)

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- Test Set Performance ---\n")
    f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}\n")

with open(os.path.join(output_dir, "cv_metrics.txt"), "w") as f:
    f.write("=== 5-Fold Cross-Validation Results (SVR) ===\n")
    for metric, stats in cv_results.items():
        f.write(f"\n{metric}:\n")
        f.write("  Individual Scores: " + ", ".join(map(str, stats['folds'])) + "\n")
        f.write(f"  Mean = {stats['mean']} | Std = {stats['std']}\n")

print(f"SVR model training complete. Outputs saved to {output_dir}/")