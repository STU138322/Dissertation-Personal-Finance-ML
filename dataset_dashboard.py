import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db_connect import load_data

# --- Page Configuration ---
st.set_page_config(page_title="Personal Finance Analysis Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview",
    "Correlation Heatmap",
    "Feature Distributions",
    "Log1p Distributions",
    "Category Analysis",
    "Category_Encoded Analysis",
    "Hypothesis Test Results"
])

# Actual database table names
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ("savings_data_blindtest", "savings_data_train")
)

# Load Data
df = load_data(dataset_choice)

# Friendly display name
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
    corr = df[['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings']].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig, use_container_width=True)

elif page == "Feature Distributions":
    feature = st.selectbox("Select Feature to Plot", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings', 'Savings_Rate'])
    selected_category = st.selectbox("Filter by Category (optional)", ['All'] + sorted(df['Category'].dropna().unique()))

    if selected_category != 'All':
        filtered_df = df[df['Category'] == selected_category]
    else:
        filtered_df = df

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df[feature], kde=True, bins=30, ax=ax)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{feature} Distribution - {friendly_name[dataset_choice]}", fontsize=16)
    st.pyplot(fig, use_container_width=True)

elif page == "Log1p Distributions":
    feature = st.selectbox("Select Feature to Log1p Transform", ['Income', 'Expense', 'Net_Savings', 'Rolling_Income', 'Rolling_Expense', 'Rolling_Savings'])
    selected_category = st.selectbox("Filter by Category (optional)", ['All'] + sorted(df['Category'].dropna().unique()))

    if selected_category != 'All':
        filtered_df = df[df['Category'] == selected_category]
    else:
        filtered_df = df

    transformed = filtered_df[feature].dropna()
    if (transformed <= 0).any():
        transformed = np.log1p(np.abs(transformed))
    else:
        transformed = np.log1p(transformed)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(transformed, kde=True, bins=30, ax=ax)
    ax.set_xlabel(f"Log1p({feature})", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Log1p {feature} Distribution - {friendly_name[dataset_choice]}", fontsize=16)
    st.pyplot(fig, use_container_width=True)

elif page == "Category Analysis":
    category_counts = df['Category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title(f"Category Distribution - {friendly_name[dataset_choice]}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig, use_container_width=True)

elif page == "Category_Encoded Analysis":
    encoded_counts = df['Category_Encoded'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=encoded_counts.index, y=encoded_counts.values, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Encoded Category")
    ax.set_title(f"Encoded Category Distribution - {friendly_name[dataset_choice]}")
    st.pyplot(fig, use_container_width=True)

elif page == "Hypothesis Test Results":
    st.write("Hypothesis testing results will be displayed here (coming soon!).")

