# combined_dashboard.py
import streamlit as st

# Shared imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro

# Project setup
from db.db_connect import load_data, FEATURES, TARGET

# === PAGE SETUP ===
st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")

# === MAIN SIDEBAR ===
main_section = st.sidebar.radio("Select Dashboard Section", ["Dataset Explorer", "Model Visualisation"])
dataset_choice = st.sidebar.selectbox("Select Dataset", ["savings_data_train", "savings_data_blindtest"])
friendly_name = {
    "savings_data_blindtest": "Dataset 1 (Blindtest)",
    "savings_data_train": "Dataset 2 (Training)"
}
df = load_data(dataset_choice)

# =============================
# SECTION 1: DATASET EXPLORER
# =============================
if main_section == "Dataset Explorer":
    st.title("Dataset Explorer")
    page = st.sidebar.radio("Analysis Tabs", [
        "Overview",
        "Correlation Heatmap",
        "Feature Distributions",
        "Log1p Distributions",
        "Category Insights",
        "Hypothesis Test Results"
    ])
    st.subheader(f"{friendly_name[dataset_choice]}")

    if page == "Overview":
        st.dataframe(df)

    elif page == "Correlation Heatmap":
        corr = df[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate']].corr()
        heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
        heatmap.update_layout(title="Correlation Heatmap", width=600, height=500)
        st.plotly_chart(heatmap, use_container_width=True)

    elif page == "Feature Distributions":
        st.markdown("> Seaborn is used for visualizing KDE (skewness).")
        feature = st.selectbox("Select Feature", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
        filtered_df = df
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df[feature], kde=True, bins=30, ax=ax)
        ax.set_title(f"{feature} Distribution - {friendly_name[dataset_choice]}")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)

    elif page == "Log1p Distributions":
        st.markdown("> Seaborn KDE used for log-transformed values.")
        feature = st.selectbox("Select Feature", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
        values = df[feature].dropna()
        transformed = np.log1p(np.abs(values)) if (values <= 0).any() else np.log1p(values)
        try:
            p_orig = shapiro(values)[1]
            p_log = shapiro(transformed)[1]
        except Exception:
            p_orig, p_log = np.nan, np.nan
        st.markdown(f"**Shapiro-Wilk Test (p-values)**\n- Original: `{p_orig:.4e}`\n- Log1p: `{p_log:.4e}`")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(transformed, kde=True, bins=30, ax=ax)
        ax.set_xlabel(f"log1p({feature})")
        ax.set_title(f"Log1p {feature} Distribution")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)

    elif page == "Category Insights":
        option = st.selectbox("Select View", ["Raw Category Count", "Encoded Category Distribution"])
        col1, col2 = st.columns([2, 1])
        with col1:
            if option == "Raw Category Count":
                category_counts = df['Category'].value_counts().reset_index()
                fig = px.bar(category_counts, x='index', y='Category', title="Raw Category Distribution")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                encoded_counts = df['Category_Encoded'].value_counts().reset_index()
                fig = px.bar(encoded_counts, x='index', y='Category_Encoded', title="Encoded Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "Category" in df.columns and "Category_Encoded" in df.columns:
                le = LabelEncoder()
                labels = df["Category"].fillna("Unknown")
                encoded = le.fit_transform(labels)
                legend_df = pd.DataFrame({"Encoded Value": encoded, "Category": labels}).drop_duplicates().sort_values("Encoded Value")
                st.markdown("**Encoded Legend:**")
                st.dataframe(legend_df)

    elif page == "Hypothesis Test Results":
        try:
            with open("notebooks/hypothesis_tests/correlation_results.txt", "r") as file:
                raw_text = file.read()
        except FileNotFoundError:
            st.error("Correlation results file not found.")
            st.stop()
        sections = raw_text.split("=== ")
        parsed = {s.split(" ===")[0]: s for s in sections if "===" in s}
        target = "Correlation Tests for Dataset2 (Training Set)" if dataset_choice == "savings_data_train" else "Correlation Tests for Dataset1 (Blindtest Set)"
        if target in parsed:
            st.code("=== " + parsed[target], language="text")
        else:
            st.warning("No correlation results found for selected dataset.")

# =============================
# SECTION 2: MODEL VISUALISATION
# =============================
elif main_section == "Model Visualisation":
    st.title("Model Visualisation")

    view = st.sidebar.radio("Select Model Tab", [
        "Growth Tracking",
        "Cross-Validation Metrics",
        "Model Metric Summary",
        "Actual vs Predicted"
    ])

    dataset_path_map = {
        "savings_data_train": "predicted_savings_tracking",
        "savings_data_blindtest": "predicted_savings_tracking_blindtest"
    }

    if view == "Growth Tracking":
        st.subheader("Growth Tracking Charts")
        model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])

        is_blindtest = "blindtest" in dataset_choice
        if is_blindtest:
            source_filter = st.multiselect("Filter by Source Type", ["original", "synthetic"])
        else:
            source_filter = []

        folder_key = dataset_path_map[dataset_choice]
        model_folder = f"{model}_blindtest" if is_blindtest else model
        full_folder = os.path.join(folder_key, model_folder)
        file_path = os.path.join(full_folder, "plot_ready.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["Date"])
            if source_filter and "Source" in df.columns:
                df = df[df["Source"].isin(source_filter)]

            min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
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
            fig.update_layout(title=f"{model.title()} - {friendly_name[dataset_choice]}", xaxis_title="Date", yaxis_title="Savings Rate")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No plot_ready.csv file found.")

    elif view == "Cross-Validation Metrics":
        st.subheader("K-Fold Validation Results")
        model = st.selectbox("Select Model", ["linear_regression", "random_forest", "svr"])
        cv_file = f"outputs/{model}/cv_metrics.txt"
        if os.path.exists(cv_file):
            metrics = {"MAE": [], "RMSE": [], "R2": []}
            with open(cv_file, "r") as f:
                for line in f:
                    try:
                        mae, rmse, r2 = map(float, line.strip().split(","))
                        metrics["MAE"].append(mae)
                        metrics["RMSE"].append(rmse)
                        metrics["R2"].append(r2)
                    except:
                        continue
            metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
            values = metrics[metric]
            fig = px.line(x=list(range(1, len(values)+1)), y=values, markers=True, title=f"{metric} Across Folds")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("cv_metrics.txt not found.")

    elif view == "Model Metric Summary":
        st.subheader("Model Metric Comparison")
        path = "outputs/model_comparison_summary.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.dataframe(df)
            metric = st.selectbox("Select Metric", ["MAE", "RMSE", "R2"])
            chart_df = df.pivot_table(index="Model", columns="Dataset", values=metric).reset_index().melt(id_vars="Model", var_name="Dataset", value_name=metric)
            fig = px.bar(chart_df, x="Model", y=metric, color="Dataset", barmode="group", title=f"{metric} by Model and Dataset")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("model_comparison_summary.csv not found.")

    elif view == "Actual vs Predicted":
        st.subheader("Actual vs Predicted Scatter Plot")

        model_map = {
            "savings_data_train": ["linear_regression", "random_forest", "svr"],
            "savings_data_blindtest": ["linear_regression_blindtest", "random_forest_blindtest", "svr_blindtest"]
        }
        model = st.selectbox("Select Model", model_map[dataset_choice])
        model_path = f"outputs/{model}"
        pred_file = "predictions.csv" if "train" in dataset_choice else "predictions_segmented.csv"
        csv_path = os.path.join(model_path, pred_file)
        fallback_img = "actual_vs_predicted.png" if "train" in dataset_choice else "actual_vs_predicted_colored.png"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            header_map = {
                "Actual Savings Rate": "Actual", "Predicted Savings Rate": "Predicted"
            }
            df.rename(columns=header_map, inplace=True)
            if "Actual" in df.columns and "Predicted" in df.columns:
                df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
                df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
                df.dropna(subset=["Actual", "Predicted"], inplace=True)
                if not df.empty:
                    fig = px.scatter(df, x="Actual", y="Predicted", trendline="ols", title=f"{model} Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("All values were non-numeric.")
            else:
                st.warning("Expected 'Actual' and 'Predicted' columns not found.")
        elif os.path.exists(os.path.join(model_path, fallback_img)):
            st.image(Image.open(os.path.join(model_path, fallback_img)), caption="Fallback Chart")
        else:
            st.warning("No prediction data or fallback plot available.")