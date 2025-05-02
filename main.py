import argparse
import subprocess

def run_command(desc, command):
    print(f"\n{desc}...")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Failed: {desc}")
    else:
        print(f"Completed: {desc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="savings prediction pipeline.")
    
    # === CORE FLAGS ===
    parser.add_argument("--train", action="store_true", help="Train models on Dataset2")
    parser.add_argument("--blindtest", action="store_true", help="Evaluate models on Dataset1 (blindtest)")
    parser.add_argument("--benchmarks", action="store_true", help="Generate benchmark summary charts")
    parser.add_argument("--growth", action="store_true", help="Run user growth tracker for Dataset2")
    parser.add_argument("--summary", action="store_true", help="Compare metrics and create visual charts")

    # === PREPROCESSING FLAGS ===
    parser.add_argument("--clean", action="store_true", help="Clean raw datasets (Dataset1 and Dataset2)")
    parser.add_argument("--synthesize", action="store_true", help="Add synthetic records to Dataset1")
    parser.add_argument("--feature", action="store_true", help="Run Feature Engineering and EDA for both datasets")
    parser.add_argument("--hypothesis", action="store_true", help="Run Pearson/Spearman Hypothesis Testing")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit Interactive Dashboard")
    parser.add_argument("--dashboard-models", action="store_true", help="Launch Streamlit Model Charts Dashboard")

    args = parser.parse_args()

    # === CLEANING RAW DATA ===
    if args.clean:
        run_command("Cleaning Raw Datasets", ["python", "scripts/clean_data.py"])

    # === SYNTHESIZING RECORDS ===
    if args.synthesize:
        run_command("Adding Synthetic Records to Dataset1", ["python", "scripts/add_records_dataset1.py"])

    # === FEATURE ENGINEERING + EDA ===
    if args.feature:
        run_command("Running Feature Engineering and EDA", ["python", "scripts/dataset1_2_feature_engineering_EDA.py"])

    # === TRAINING MODELS ===
    if args.train:
        run_command("Training Linear Regression", ["python", "models/linear_model_dataset2.py"])
        run_command("Training Random Forest", ["python", "models/random_forest_model_dataset2.py"])
        run_command("Training SVR", ["python", "models/svr_model_dataset2.py"])

    # === EVALUATING ON BLINDTEST DATA ===
    if args.blindtest:
        run_command("Running Linear Regression Blindtest", ["python", "models/linear_regression_blindtest_dataset1.py"])
        run_command("Running Random Forest Blindtest", ["python", "models/random_forest_blindtest_dataset1.py"])
        run_command("Running SVR Blindtest", ["python", "models/svr_blindtest_dataset1.py"])

    # === GENERATE BENCHMARK THRESHOLD CHARTS ===
    if args.benchmarks:
        run_command("Generating Benchmark Summaries", ["python", "scripts/benchmark_summary.py"])
        run_command("Generating Blindtest Benchmark Chart", ["python", "scripts/benchmark_blindtest_chart_generation.py"])
        run_command("Generating Dataset2 Benchmark Chart", ["python", "scripts/benchmark_dataset2_chart_generation.py"])

    # === TRACK GROWTH OVER TIME (Dataset1 + Dataset2) ===
    if args.growth:
        run_command("Running Growth Tracker on Dataset2", ["python", "scripts/growth_tracker.py", "--all"])
        run_command("Running Growth Tracker on Dataset1 (Blindtest)", ["python", "scripts/growth_tracker.py", "--blindtest"])

    # === METRIC COMPARISON & CHARTS ===
    if args.summary:
        run_command("Generating Model Comparison Summary", ["python", "scripts/compare_metrics.py"])
        run_command("Creating Metric Comparison Charts", ["python", "scripts/metric_chart_creation.py"])

    # === HYPOTHESIS TESTING ===
    if args.hypothesis:
        run_command("Running Hypothesis Testing", ["python", "scripts/pearson_spearman_testing.py"])

    # === STREAMLIT DASHBOARD ===
    if args.dashboard:
        run_command("Launching Streamlit Dashboard", ["streamlit", "run", "app_dashboard.py"])
    if args.dashboard_models:
        run_command("Launching Model Charts Dashboard", ["streamlit", "run", "models_dashboard.py"])

    # === DEFAULT MESSAGE ===
    if not any(vars(args).values()):
        print("\nNo actions specified. Use flags like --clean, --synthesize, --feature, --train, --hypothesis, --dashboard etc.")
