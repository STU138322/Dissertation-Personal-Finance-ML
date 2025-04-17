import os
import shutil
import glob

EXPORT_DIR = "submission_export"

# === File groups for focused export ===
EXPORT_PATHS = {
    "benchmarks": [
        "benchmarks/plot_blindtest_threshold_summary.png",
        "benchmarks/plot_dataset2_threshold_summary.png"
    ],
    "charts": glob.glob("outputs/charts_all_data/*.png"),
    "growth": [
        "outputs/linear_regression_growth_tracker/user_growth_tracking.csv",
        "outputs/decision_tree_growth_tracker/user_growth_tracking.csv",
        "outputs/random_forest_growth_tracker/user_growth_tracking.csv"
    ],
    "summary": [
        "outputs/model_comparison_summary.csv"
    ],
    "eda/processed": glob.glob("notebooks/processed_EDA/*.png"),
    "eda/engineered": glob.glob("notebooks/engineered_EDA/*.png")
}

def export_group(group_name, files):
    group_path = os.path.join(EXPORT_DIR, group_name)
    os.makedirs(group_path, exist_ok=True)

    for f in files:
        if os.path.exists(f):
            shutil.copy2(f, group_path)
            print(f"Copied: {f} → {group_path}")
        else:
            print(f"Missing: {f}")

def copy_full_folder(source, dest_name):
    dest_path = os.path.join(EXPORT_DIR, dest_name)
    if os.path.exists(source):
        shutil.copytree(source, dest_path)
        print(f"Full folder copied: {source} → {dest_path}")
    else:
        print(f"Folder not found: {source}")

if __name__ == "__main__":
    print(f"\nExporting key results to: {EXPORT_DIR}")
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)

    # Export selected file groups
    for group, files in EXPORT_PATHS.items():
        print(f"\nExporting group: {group}")
        export_group(group, files)

    # Copy entire outputs and benchmarks folder
    print("\nCopying full folders...")
    copy_full_folder("outputs", "full_outputs")
    copy_full_folder("benchmarks", "full_benchmarks")

    print("\nReport export complete! Deliverables saved in /submission_export/")
