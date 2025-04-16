import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load engineered dataset2 (main dataset)
file_path = 'data/engineered/engineered_dataset2.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

# Drop any remaining NaNs
df = df.dropna()

# Select features and target for training
features = ['Income', 'Expense', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']
target = 'Net_Savings'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Linear Regression Evaluation on Dataset 2 ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Create subfolder for linear regression results
output_dir = 'outputs/linear_regression'
os.makedirs(output_dir, exist_ok=True)

# Plot actual vs predicted savings
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Net Savings')
plt.ylabel('Predicted Net Savings')
plt.title('Actual vs Predicted Net Savings (Linear Regression - Dataset 2)')
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
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n")

print(f"Linear Regression model training complete. Results saved to {output_dir}/")
