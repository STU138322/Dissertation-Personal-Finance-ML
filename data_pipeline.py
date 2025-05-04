import argparse
import subprocess

def run_command(description, command):
    print(f"\n{description}...")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Failed: {description}")
    else:
        print(f"Completed: {description}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data preparation pipeline.")
    parser.add_argument("--clean", action="store_true", help="Clean raw CSVs")
    parser.add_argument("--augment", action="store_true", help="Add synthetic records to Dataset1")
    parser.add_argument("--engineer", action="store_true", help="Feature engineer Dataset1 & Dataset2")
    parser.add_argument("--load", action="store_true", help="Load engineered data into SQLite DB")
    parser.add_argument("--all", action="store_true", help="Run full data pipeline in order")

    args = parser.parse_args()

    if args.all or args.clean:
        run_command("Cleaning raw CSVs", ["python", "scripts/clean_data.py"])

    if args.all or args.augment:
        run_command("Generating synthetic records (Dataset1)", ["python", "scripts/add_records_dataset1.py"])

    if args.all or args.engineer:
        run_command("Feature engineering: Dataset1&2", ["python", "scripts/dataset1_2_feature_engineering_EDA.py"])

    if args.all or args.load:
        run_command("Loading data into SQLite DB", ["python", "db/sqlite_db_integration.py"])

    if not any(vars(args).values()):
        print("\nNo pipeline stage selected. Use --clean, --augment, --engineer, --load, or --all.")
