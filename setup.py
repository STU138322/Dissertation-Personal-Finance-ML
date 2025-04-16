from pathlib import Path

# .gitignore content
gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
env/
venv/
.ipynb_checkpoints/
.vscode/
.DS_Store
Thumbs.db
*.csv
*.sqlite
*.db
*.h5
*.pkl
*.joblib
outputs/
"""

# README.md content
readme_content = """
# Dissertation-Personal-Finance-ML

STU138322 – Dissertation Project – Personal Finance Forecasting Using Machine Learning

## 📘 Project Overview
This project explores the use of machine learning models to forecast personal net savings using financial history (income, expenses, etc.). Models are trained and evaluated to determine which best predicts savings growth.

## 📂 Project Structure

- `/data` – Raw and cleaned datasets used for training and testing
- `/models` – Trained ML models saved for reuse
- `/notebooks` – Jupyter notebooks for EDA and experimentation
- `/outputs` – Generated charts, predictions, evaluation metrics
- `/scripts` – Python scripts for training, testing, and evaluation
- `main.py` – Main orchestrator script
- `requirements.txt` – List of Python dependencies

## ⚙️ Getting Started

```bash
pip install -r requirements.txt
python main.py

📊 Models Used
Linear Regression

Random Forest

Decision Tree

(Optional) XGBoost

📈 Evaluation Metrics
MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

🧠 Author
Sander (STU138322)
Final Year BSc Computing, Arden University """

#requirements.txt content
requirements_content = """ pandas numpy matplotlib scikit-learn xgboost jupyter """

#Save the files
Path(".gitignore").write_text(gitignore_content.strip(), encoding="utf-8")
Path("README.md").write_text(readme_content.strip(), encoding="utf-8")
Path("requirements.txt").write_text(requirements_content.strip(), encoding="utf-8")
print("✅ .gitignore, README.md, and requirements.txt generated successfully!")