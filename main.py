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
    
    parser.add_argument("--train", action="store_true", help="Train models on Dataset2")
    parser.add_argument("--blindtest", action="store_true", help="Evaluate models on Dataset1 (blindtest)")
    parser.add_argument("--benchmarks", action="store_true", help="Generate benchmark summary charts")
    parser.add_argument("--growth", action="store_true", help="Run user growth tracker for Dataset2")
    parser.add_argument("--summary", action="store_true", help="Compare metrics and create visual charts")

    args = parser.parse_args()

    # === TRAINING MODELS ===
    if args.train:
        run_command("Training Linear Regression", ["python", "models/linear_model_dataset2.py"])
        run_command("Training Decision Tree", ["python", "models/decision_tree_model_dataset2.py"])
        run_command("Training Random Forest", ["python", "models/random_forest_model_dataset2.py"])

    # === EVALUATING ON BLINDTEST DATA ===
    if args.blindtest:
        run_command("Running Linear Regression Blindtest", ["python", "models/linear_regression_blindtest_dataset1.py"])
        run_command("Running Random Forest Blindtest", ["python", "models/random_forest_blindtest_dataset1.py"])
        run_command("Running Decision Tree Blindtest", ["python", "models/decision_tree_blindtest_dataset1.py"])

    # === GENERATE BENCHMARK THRESHOLD CHARTS ===
    if args.benchmarks:
        run_command("Generating Blindtest Benchmark Chart", ["python", "scripts/benchmark_blindtest_chart_generation.py"])
        run_command("Generating Dataset2 Benchmark Chart", ["python", "scripts/benchmark_dataset2_chart_generation.py"])

    # === TRACK GROWTH OVER TIME (TRAINING SET) ===
    if args.growth:
        run_command("Running Growth Tracker on Dataset2", ["python", "scripts/growth_tracker.py", "--all"])

    # === METRIC COMPARISON & CHARTS ===
    if args.summary:
        run_command("Generating Model Comparison Summary", ["python", "scripts/compare_metrics.py"])
        run_command("Creating Metric Comparison Charts", ["python", "scripts/metric_chart_creation.py"])

    # === DEFAULT MESSAGE ===
    if not any(vars(args).values()):
        print("\nNo actions specified. Use one or more flags: --train, --blindtest, --benchmarks, --growth, --summary")
