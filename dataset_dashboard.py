import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data

st.set_page_config(page_title="Personal Finance Analysis Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview",
    "Correlation Heatmap",
    "Feature Distributions",
    "Log1p Distributions",
    "Category Insights",
    "Hypothesis Test Results"
])

dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ("savings_data_blindtest", "savings_data_train")
)

df = load_data(dataset_choice)

friendly_name = {
    "savings_data_blindtest": "Dataset 1 (Blindtest)",
    "savings_data_train": "Dataset 2 (Training)"
}

st.title("Personal Finance Analysis Dashboard")
st.subheader(f"Currently viewing: {friendly_name[dataset_choice]}")
st.write(f"Total records: {len(df)}")

# --- Pages ---
if page == "Overview":
    st.dataframe(df)

elif page == "Correlation Heatmap":
    corr = df[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate']].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    fig.tight_layout(pad=1.0)
    st.pyplot(fig)

elif page == "Feature Distributions":
    feature = st.selectbox("Select Feature to Plot", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
    selected_category = st.selectbox("Filter by Category (optional)", ['All'] + sorted(df['Category'].dropna().unique()))
    filtered_df = df if selected_category == 'All' else df[df['Category'] == selected_category]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df[feature], kde=True, bins=30, ax=ax)
    ax.set_title(f"{feature} Distribution - {friendly_name[dataset_choice]}")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig)

elif page == "Log1p Distributions":
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
    ax.set_title(f"Log1p {feature} Distribution - {friendly_name[dataset_choice]}")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig)

elif page == "Category Insights":
    st.subheader("Category Insights Viewer")

    option = st.selectbox("Select View", ["Raw Category Count", "Encoded Category Distribution"])
    col1, col2 = st.columns([2, 1])

    with col1:
        if option == "Raw Category Count":
            category_counts = df['Category'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title(f"Category Distribution - {friendly_name[dataset_choice]}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)
        else:
            encoded_counts = df['Category_Encoded'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=encoded_counts.index, y=encoded_counts.values, ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Encoded Category")
            ax.set_title(f"Encoded Category Distribution - {friendly_name[dataset_choice]}")
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)

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
