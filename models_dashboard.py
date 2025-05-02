import streamlit as st
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Charts and Benchmarks", layout="wide")

st.title("Model Visualisation Dashboard")
st.markdown("Browse growth tracking plots, benchmark results, and metric comparisons for all models.")

st.sidebar.title("Navigation")
view = st.sidebar.radio("Select View", [
    "Growth Tracking Charts",
    "Benchmark Charts",
    "Model Metric Comparison",
    "All Available Charts",
    "Blindtest Comparison Charts",
    "Actual vs Predicted Charts"
])

def show_images_from_folder(folder_path, title_prefix, filters=None):
    if not os.path.exists(folder_path):
        st.warning(f"No folder found at {folder_path}")
        return
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".png") and (not filters or any(filt.lower() in f.lower() for filt in filters))
    ])
    if not image_files:
        st.info("No chart images available.")
        return
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        st.subheader(f"{title_prefix} - {img_file}")
        st.image(Image.open(img_path), use_container_width=True)

if view == "Growth Tracking Charts":
    base_dirs = {
        "Dataset2 - Training": "predicted_savings_tracking",
        "Dataset1 - Blindtest": "predicted_savings_tracking_blindtest"
    }
    selected_dataset = st.selectbox("Select Dataset", list(base_dirs.keys()))
    model = st.selectbox("Model", ["linear_regression", "random_forest", "svr"])
    source_filter = st.multiselect("Filter by Source Type (optional)", ["original", "synthetic"])
    category_filter = st.text_input("Filter by Category Keyword (optional)")

    model_folder = model
    if selected_dataset == "Dataset1 - Blindtest":
        model_folder = f"{model}_blindtest"

    folder_path = os.path.join(base_dirs[selected_dataset], model_folder)
    filters = source_filter + ([category_filter] if category_filter else [])
    show_images_from_folder(folder_path, f"{selected_dataset} - {model.replace('_', ' ').title()}", filters)

elif view == "Benchmark Charts":
    st.title("Benchmark Threshold Charts")
    show_images_from_folder("benchmarks", "Benchmark")

elif view == "Model Metric Comparison":
    st.title("Model Metric Summary")

    metric_path = os.path.join("outputs", "model_comparison_summary.csv")
    if os.path.exists(metric_path):
        df = pd.read_csv(metric_path)
        st.dataframe(df)

        metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
        chart_df = df.pivot_table(index="Model", columns="Dataset", values=metric, aggfunc="mean").reset_index()

        st.subheader(f"{metric} Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        chart_df.set_index("Model").plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Model and Dataset")
        ax.legend(title="Dataset")
        st.pyplot(fig)
    else:
        st.warning("Model comparison summary file not found. Run `python main.py --summary` to generate it.")

elif view == "All Available Charts":
    st.subheader("Charts in Benchmarks Folder")
    show_images_from_folder("benchmarks", "Benchmarks")

    st.subheader("Top-Level Charts in Outputs Folder")
    show_images_from_folder("outputs", "outputs")

    st.subheader("Charts in Outputs Subfolders")
    for subfolder in os.listdir("outputs"):
        full_path = os.path.join("outputs", subfolder)
        if os.path.isdir(full_path):
            show_images_from_folder(full_path, f"outputs/{subfolder}")

    st.subheader("Predicted Savings Tracking (Dataset2)")
    for subfolder in os.listdir("predicted_savings_tracking"):
        full_path = os.path.join("predicted_savings_tracking", subfolder)
        if os.path.isdir(full_path):
            show_images_from_folder(full_path, f"predicted_savings_tracking/{subfolder}")

    st.subheader("Predicted Savings Tracking Blindtest (Dataset1)")
    for subfolder in os.listdir("predicted_savings_tracking_blindtest"):
        full_path = os.path.join("predicted_savings_tracking_blindtest", subfolder)
        if os.path.isdir(full_path):
            show_images_from_folder(full_path, f"predicted_savings_tracking_blindtest/{subfolder}")

elif view == "Blindtest Comparison Charts":
    st.subheader("Charts for Dataset1 Blindtest Comparison")
    for folder in ["charts_all_data", "charts_original_only", "charts_synthetic_only"]:
        folder_path = os.path.join("outputs", folder)
        show_images_from_folder(folder_path, folder)

elif view == "Actual vs Predicted Charts":
    st.subheader("Actual vs Predicted Model Charts")
    model_folders = [
        "linear_regression", "random_forest", "svr",
        "linear_regression_blindtest", "random_forest_blindtest", "svr_blindtest"
    ]
    selected_model = st.selectbox("Select Model Folder", model_folders)
    model_path = os.path.join("outputs", selected_model)
    show_images_from_folder(model_path, selected_model)
