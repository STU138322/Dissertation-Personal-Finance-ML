import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data

st.set_page_config(page_title="Personal Finance Analysis Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview",
    "Correlation Heatmap",
    "Feature Distributions",
    "Log1p Distributions",
    "Category Insights",
    "Hypothesis Test Results"
])

dataset_choice = st.sidebar.selectbox("Select Dataset", ["savings_data_blindtest", "savings_data_train"])
df = load_data(dataset_choice)

friendly_name = {
    "savings_data_blindtest": "Dataset 1 (Blindtest)",
    "savings_data_train": "Dataset 2 (Training)"
}

st.title("Personal Finance Analysis Dashboard")
st.subheader(f"Currently viewing: {friendly_name[dataset_choice]}")
st.write(f"Total records: {len(df)}")

# === Pages ===

if page == "Overview":
    st.dataframe(df)

elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    corr = df[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate']].corr()
    heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    heatmap.update_layout(title="Correlation Heatmap", width=600, height=500)
    st.plotly_chart(heatmap, use_container_width=True)

elif page == "Feature Distributions":
    st.subheader("Feature Distributions")
    st.markdown("> _This plot uses Seaborn to display a KDE (smoothed density curve) which helps visualize skewness. Plotly currently does not support KDE natively._")
    feature = st.selectbox("Select Feature", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
    selected_category = st.selectbox("Filter by Category (optional)", ['All'] + sorted(df['Category'].dropna().unique()))
    filtered_df = df if selected_category == 'All' else df[df['Category'] == selected_category]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df[feature], kde=True, bins=30, ax=ax)
    ax.set_title(f"{feature} Distribution - {friendly_name[dataset_choice]}")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig)

elif page == "Log1p Distributions":
    st.subheader("Log1p Distributions")
    st.markdown("> _This plot uses Seaborn to show skewness after log transformation. KDE lines make distribution tails easier to interpret._")
    feature = st.selectbox("Select Feature to Log1p Transform", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
    selected_category = st.selectbox("Filter by Category (optional)", ['All'] + sorted(df['Category'].dropna().unique()))
    filtered_df = df if selected_category == 'All' else df[df['Category'] == selected_category]

    values = filtered_df[feature].dropna()
    if len(values) < 3:
        st.warning("Not enough data to compute normality test.")
        st.stop()

    transformed = np.log1p(np.abs(values)) if (values <= 0).any() else np.log1p(values)

    try:
        p_orig = shapiro(values)[1]
        p_log = shapiro(transformed)[1]
    except Exception:
        p_orig, p_log = np.nan, np.nan

    st.markdown("**Shapiro-Wilk Test (Normality):**")
    st.markdown(f"- Original p-value: `{p_orig:.4e}`")
    st.markdown(f"- Log1p p-value: `{p_log:.4e}`")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(transformed, kde=True, bins=30, ax=ax)
    ax.set_xlabel(f"log1p({feature})")
    ax.set_title(f"Log1p {feature} Distribution - {friendly_name[dataset_choice]}")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig)

elif page == "Category Insights":
    st.subheader("Category Insights Viewer")

    option = st.selectbox("Select View", ["Raw Category Count", "Encoded Category Distribution"])
    col1, col2 = st.columns([2, 1])

    with col1:
        if option == "Raw Category Count":
            category_counts = df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            fig = px.bar(category_counts, x='Category', y='Count', title="Raw Category Distribution")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            encoded_counts = df['Category_Encoded'].value_counts().reset_index()
            encoded_counts.columns = ['Encoded', 'Count']
            fig = px.bar(encoded_counts, x='Encoded', y='Count', title="Encoded Category Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Category" in df.columns and "Category_Encoded" in df.columns:
            le = LabelEncoder()
            labels = df["Category"].fillna("Unknown")
            encoded = le.fit_transform(labels)
            legend_df = pd.DataFrame({"Encoded Value": encoded, "Category": labels})
            legend_df = legend_df.drop_duplicates().sort_values("Encoded Value")
            st.markdown("**Encoded Legend:**")
            st.dataframe(legend_df)
        else:
            st.warning("Encoding legend not available.")

elif page == "Hypothesis Test Results":
    st.subheader("Hypothesis Test Results")
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
