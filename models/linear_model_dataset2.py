import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import sys

# Load project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data, FEATURES, TARGET, TABLE_TRAIN

sns.set_theme(style="whitegrid")

# Load Data
df = load_data(TABLE_TRAIN).dropna().sort_values('Date')
X = df[FEATURES]
y = df[TARGET]

# 70:30 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression Model with RobustScaler
pipeline = make_pipeline(RobustScaler(), LinearRegression())
pipeline.fit(X_train, y_train)

# Save Model Pipeline
model_dir = 'models/linear_regression'
os.makedirs(model_dir, exist_ok=True)
dump(pipeline, os.path.join(model_dir, 'model.pkl'))

# Predict on Test Set
y_pred = pipeline.predict(X_test)

# Prepare Predictions with Date and Category
test_results = df.loc[X_test.index, ["Date", "Category"]].copy()
test_results["Actual"] = y_test.values
test_results["Predicted"] = y_pred

# Evaluate Model
mae = mean_absolute_error(test_results["Actual"], test_results["Predicted"])
rmse = np.sqrt(mean_squared_error(test_results["Actual"], test_results["Predicted"]))
r2 = r2_score(test_results["Actual"], test_results["Predicted"])

print("\n--- Linear Regression Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Cross-Validation (5-Fold)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    'MAE': make_scorer(mean_absolute_error),
    'RMSE': make_scorer(rmse_score),
    'R2': make_scorer(r2_score)
}

cv_results = {}
for name, scorer in scorers.items():
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer)
    rounded_scores = [round(s, 2) for s in scores]
    cv_results[name] = {
        'folds': rounded_scores,
        'mean': round(np.mean(scores), 2),
        'std': round(np.std(scores), 2)
    }

# Save Outputs 
output_dir = 'outputs/linear_regression'
os.makedirs(output_dir, exist_ok=True)

# Save Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=test_results["Actual"], y=test_results["Predicted"])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Linear Regression - Dataset 2)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

# Save Test Predictions CSV
cols_to_save = ["Date", "Category", "Actual", "Predicted"]
test_results[cols_to_save].to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save Test Metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("--- Test Set Performance ---\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.2f}\n")

# Save Cross-Validation Metrics
with open(os.path.join(output_dir, "cv_metrics.txt"), "w") as f:
    for i in range(5):
        mae_fold = cv_results["MAE"]["folds"][i]
        rmse_fold = cv_results["RMSE"]["folds"][i]
        r2_fold = cv_results["R2"]["folds"][i]
        f.write(f"{mae_fold},{rmse_fold},{r2_fold}\n")

print(f"\nLinear Regression model training complete. Outputs saved to {output_dir}/")
