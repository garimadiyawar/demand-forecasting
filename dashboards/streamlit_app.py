import streamlit as st
import pandas as pd

st.title("Demand Forecasting Dashboard")

df = pd.read_parquet("data/processed/features.parquet")

sku = st.selectbox("Select SKU", df["id"].unique())
subset = df[df["id"] == sku]

st.line_chart(subset.set_index("date")[["sales"]])
