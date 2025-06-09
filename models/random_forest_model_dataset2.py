import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import os
import sys

# Load project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# Load and prepare data
df = load_data(TABLE_TRAIN).dropna().sort_values("Date")
X = df[FEATURES]
y = df[TARGET]

# Time-based split (70/30)
split_index = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train model with RobustScaler + tuned hyperparameters
pipeline = make_pipeline(
    RobustScaler(),
    RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
)
pipeline.fit(X_train, y_train)

# Save model
model_dir = 'models/random_forest'
os.makedirs(model_dir, exist_ok=True)
dump(pipeline, os.path.join(model_dir, 'model.pkl'))

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Random Forest Evaluation on Dataset 2 (RobustScaler + Tuned) ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Cross-validation
def rmse_score(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    'MAE': make_scorer(mean_absolute_error),
    'RMSE': make_scorer(rmse_score),
    'R2': make_scorer(r2_score)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, scorer in scorers.items():
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer)
    cv_results[name] = {
        'folds': [round(s, 2) for s in scores],
        'mean': round(np.mean(scores), 2),
        'std': round(np.std(scores), 2)
    }

# Save Outputs
output_dir = 'outputs/random_forest'
os.makedirs(output_dir, exist_ok=True)

# Save scatter plot (Actual vs Predicted)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Random Forest - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

# Save feature importances
importances = pipeline.named_steps['randomforestregressor'].feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=FEATURES)
plt.title("Feature Influence on Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# Save predictions CSV in standardized format
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save metrics summary
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- Test Set Performance ---\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.2f}\n")

# Save cross-validation metrics for dashboard
with open(os.path.join(output_dir, "cv_metrics.txt"), "w") as f:
    for i in range(5):
        mae = cv_results["MAE"]["folds"][i]
        rmse = cv_results["RMSE"]["folds"][i]
        r2 = cv_results["R2"]["folds"][i]
        f.write(f"{mae},{rmse},{r2}\n")

print(f"Random Forest training complete. Outputs saved to {output_dir}/")
