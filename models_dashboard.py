import streamlit as st
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Visualisation Dashboard", layout="wide")
st.title("Model Visualisation Dashboard")
st.markdown("Explore model growth tracking, predictions, and cross-validation results.")

st.sidebar.title("Navigation")
view = st.sidebar.radio("Select View", [
    "Growth Tracking",
    "Cross-Validation Metrics",
    "Model Metric Summary",
    "Actual vs Predicted"
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
        st.image(Image.open(img_path))

# === 1. Growth Tracking ===
if view == "Growth Tracking":
    st.subheader("Growth Tracking Results")
    base_dirs = {
        "Dataset2 - Training": "predicted_savings_tracking",
        "Dataset1 - Blindtest": "predicted_savings_tracking_blindtest"
    }
    dataset_choice = st.selectbox("Select Dataset", list(base_dirs.keys()))
    model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])
    source_filter = st.multiselect("Filter by Source Type (optional)", ["original", "synthetic"])
    keyword_filter = st.text_input("Filter by Category Keyword (optional)")

    model_folder = f"{model}_blindtest" if "Blindtest" in dataset_choice else model
    folder_path = os.path.join(base_dirs[dataset_choice], model_folder)
    filters = source_filter + ([keyword_filter] if keyword_filter else [])

    show_images_from_folder(folder_path, f"{dataset_choice} - {model}", filters)

# === 2. Cross-Validation Metrics (NEW) ===
elif view == "Cross-Validation Metrics":
    st.subheader("K-Fold Validation Results")
    model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])
    metric_file = f"outputs/{model}/cv_metrics.txt"

    if os.path.exists(metric_file):
        with open(metric_file, "r") as f:
            lines = f.readlines()

        metrics = {"MAE": [], "RMSE": [], "R2": []}
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    metrics["MAE"].append(float(parts[0]))
                    metrics["RMSE"].append(float(parts[1]))
                    metrics["R2"].append(float(parts[2]))
                except ValueError:
                    continue

        selected_metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
        data = metrics[selected_metric]
        if data:
            st.markdown(f"**{model.replace('_', ' ').title()} — {selected_metric}**")
            st.markdown(f"- Mean: `{pd.Series(data).mean():.4f}`")
            st.markdown(f"- Std Dev: `{pd.Series(data).std():.4f}`")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data, marker='o')
            ax.set_title(f"{selected_metric} Across K-Folds")
            ax.set_xlabel("Fold")
            ax.set_ylabel(selected_metric)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)
        else:
            st.info("No valid metric values found.")
    else:
        st.warning("Cross-validation file not found.")

# === 3. Model Metric Summary ===
elif view == "Model Metric Summary":
    st.subheader("Model Metric Comparison")
    metric_path = os.path.join("outputs", "model_comparison_summary.csv")

    if os.path.exists(metric_path):
        df = pd.read_csv(metric_path)
        st.dataframe(df)

        metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
        chart_df = df.pivot_table(index="Model", columns="Dataset", values=metric, aggfunc="mean").reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        chart_df.set_index("Model").plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Model and Dataset")
        ax.legend(title="Dataset")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)
    else:
        st.warning("Model comparison summary file not found. Run `python main.py --summary` to generate it.")

# === 4. Actual vs Predicted Charts ===
elif view == "Actual vs Predicted":
    st.subheader("Actual vs Predicted Model Outputs")
    model_folders = [
        "linear_regression", "random_forest", "svr",
        "linear_regression_blindtest", "random_forest_blindtest", "svr_blindtest"
    ]
    selected_model = st.selectbox("Select Model Folder", model_folders)
    model_path = os.path.join("outputs", selected_model)
    show_images_from_folder(model_path, selected_model)
