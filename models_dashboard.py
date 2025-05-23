import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Model Visualisation Dashboard", layout="wide")
st.title("Model Visualisation Dashboard")
st.markdown("Explore model growth tracking, predictions, and cross-validation results.")

# Sidebar
st.sidebar.title("Navigation")
selected_dataset = st.sidebar.selectbox("Select Dataset", ["Dataset2 (Training)", "Dataset1 (Blindtest)"])
view = st.sidebar.radio("Select View", [
    "Growth Tracking",
    "Cross-Validation Metrics",
    "Model Metric Summary",
    "Actual vs Predicted"
])

dataset_path_map = {
    "Dataset2 (Training)": "predicted_savings_tracking",
    "Dataset1 (Blindtest)": "predicted_savings_tracking_blindtest"
}

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
    st.subheader("Growth Tracking Charts")
    model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])

    show_source_filter = "Blindtest" in selected_dataset
    source_filter = []
    if show_source_filter:
        source_filter = st.multiselect("Filter by Source Type (optional)", ["original", "synthetic"])
    keyword_filter = st.text_input("Filter by Category Keyword (optional)")

    folder_key = dataset_path_map[selected_dataset]
    model_folder = f"{model}_blindtest" if "Blindtest" in selected_dataset else model
    full_folder = os.path.join(folder_key, model_folder)
    file_path = os.path.join(full_folder, "plot_ready.csv")

    if not os.path.exists(file_path):
        st.info("No CSV growth data found. Fallback to chart images.")
        show_images_from_folder(full_folder, model)
    else:
        df = pd.read_csv(file_path, parse_dates=["Date"])

        # Filter by keyword if needed
        if keyword_filter:
            df = df[df.apply(lambda row: row.astype(str).str.contains(keyword_filter, case=False).any(), axis=1)]

        # Apply source filter only for Dataset1
        if show_source_filter and "Source" in df.columns and source_filter:
            df = df[df["Source"].isin(source_filter)]

        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()

        st.markdown("**Filter by Time Range**")
        date_range = st.date_input("Select Date Range:", (min_date, max_date), min_value=min_date, max_value=max_date)
        if st.button("Reset Date Range"):
            date_range = (min_date, max_date)

        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

        fig = go.Figure()
        if "Actual_Savings_Rate" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual_Savings_Rate"], mode='lines', name="Actual", line=dict(color="green")))
        if "Predicted_Savings_Rate_Smoothed" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted_Savings_Rate_Smoothed"], mode='lines', name="Predicted", line=dict(color="red", dash="dash")))

        fig.update_layout(title=f"{model.replace('_', ' ').title()} - {selected_dataset}",
                          xaxis_title="Date",
                          yaxis_title="Savings Rate",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

# === 2. Cross-Validation Metrics ===
elif view == "Cross-Validation Metrics":
    st.subheader("K-Fold Validation Results")
    model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])
    cv_file = os.path.join("outputs", model, "cv_metrics.txt")

    if os.path.exists(cv_file):
        metrics = {"MAE": [], "RMSE": [], "R2": []}
        with open(cv_file, "r") as f:
            for line in f:
                if line.strip() == "" or line.startswith("#"):
                    continue
                try:
                    mae, rmse, r2 = map(float, line.strip().split(","))
                    metrics["MAE"].append(mae)
                    metrics["RMSE"].append(rmse)
                    metrics["R2"].append(r2)
                except:
                    continue

        selected_metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
        data = metrics[selected_metric]
        if data:
            st.markdown(f"**{model.replace('_', ' ').title()} — {selected_metric}**")
            st.markdown(f"- Mean: `{pd.Series(data).mean():.4f}`")
            st.markdown(f"- Std Dev: `{pd.Series(data).std():.4f}`")

            fig = px.line(
                x=list(range(1, len(data)+1)),
                y=data,
                markers=True,
                labels={"x": "Fold", "y": selected_metric},
                title=f"{selected_metric} Across K-Folds"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid metric values found.")
    else:
        st.warning("K-Fold cross-validation file not found.")

# === 3. Model Metric Summary ===
elif view == "Model Metric Summary":
    st.subheader("Model Metric Comparison")
    metric_file = os.path.join("outputs", "model_comparison_summary.csv")

    if os.path.exists(metric_file):
        df = pd.read_csv(metric_file)
        st.dataframe(df)

        metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
        chart_df = df.pivot_table(index="Model", columns="Dataset", values=metric, aggfunc="mean").reset_index()
        chart_df = chart_df.melt(id_vars="Model", var_name="Dataset", value_name=metric)

        fig = px.bar(
            chart_df,
            x="Model",
            y=metric,
            color="Dataset",
            barmode="group",
            title=f"{metric} by Model and Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model comparison summary file not found.")

# === 4. Actual vs Predicted ===
elif view == "Actual vs Predicted":
    st.subheader("Actual vs Predicted Charts")

    model_options = {
        "Dataset2 (Training)": ["linear_regression", "random_forest", "svr"],
        "Dataset1 (Blindtest)": ["linear_regression_blindtest", "random_forest_blindtest", "svr_blindtest"]
    }

    selected_model = st.selectbox("Select Model", model_options[selected_dataset])
    model_path = os.path.join("outputs", selected_model)

    # Determine file paths
    is_blindtest = "Blindtest" in selected_dataset
    csv_file = "predictions_segmented.csv" if is_blindtest else "predictions.csv"
    image_file = "actual_vs_predicted_colored.png" if is_blindtest else "actual_vs_predicted.png"

    csv_path = os.path.join(model_path, csv_file)
    image_path = os.path.join(model_path, image_file)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Flexible header detection
        col_map = {
            "Actual Savings Rate": "Actual",
            "Predicted Savings Rate": "Predicted",
            "Actual Net Savings": "Actual",
            "Predicted Net Savings": "Predicted",
        }
        df.rename(columns=col_map, inplace=True)

        # Ensure at least two usable columns
        usable_cols = [c for c in df.columns if df[c].dtype != 'object']
        if "Actual" in df.columns and "Predicted" in df.columns:
            x, y = "Actual", "Predicted"
        elif len(df.columns) >= 2:
            x, y = df.columns[0], df.columns[1]
        else:
            st.warning("Prediction file must have at least two numeric columns.")
            st.stop()

        # Convert to numeric and drop rows that cannot be converted
        df[x] = pd.to_numeric(df[x].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        df[y] = pd.to_numeric(df[y].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        df = df.dropna(subset=[x, y])

        if df.empty:
            st.warning("No valid numeric data found in prediction file.")
        else:
            fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"{x} vs {y}")
            st.plotly_chart(fig, use_container_width=True)
    elif os.path.exists(image_path):
        st.image(Image.open(image_path), caption="Fallback Image Plot")
    else:
        st.warning("No prediction results or fallback chart found for this model.")
