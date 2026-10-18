from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(
    page_title="Real Estate Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Mumbai Real Estate Insights")
st.write("Understand the main patterns and useful findings from the property dataset.")

# Load data
DATA_PATH = "data/base-file/analysis.csv"  # Change this path if needed

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# Check columns
required_cols = [
    "price", "area", "bedrooms", "bathrooms", "category",
    "furnished_status", "type_of_building", "status", "neworold"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# Prepare data
for col in ["price", "area", "bedrooms", "bathrooms"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price", "area"])
df = df[(df["price"] >= 0) & (df["area"] > 0)]

if df.empty:
    st.error("No valid property records found.")
    st.stop()

furnishing_map = {
    "0": "Unfurnished", "0.0": "Unfurnished",
    "1": "Semi-Furnished", "1.0": "Semi-Furnished",
    "2": "Fully Furnished", "2.0": "Fully Furnished"
}
df["furnishing_label"] = (
    df["furnished_status"].astype(str).str.strip().replace(furnishing_map)
)

# Filters
st.sidebar.header("Insight Filters")

regions = sorted(df["category"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region", regions, default=regions
)

property_types = sorted(df["type_of_building"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Building Type", property_types, default=property_types
)

statuses = sorted(df["status"].dropna().unique().tolist())
selected_statuses = st.sidebar.multiselect(
    "Property Status", statuses, default=statuses
)

filtered_df = df[
    df["category"].isin(selected_regions)
    & df["type_of_building"].isin(selected_types)
    & df["status"].isin(selected_statuses)
].copy()

if filtered_df.empty:
    st.warning("No properties match these filters.")
    st.stop()

# Overview
st.subheader("Market Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Properties Analysed", f"{len(filtered_df):,}")
c2.metric("Average Price", f"₹{filtered_df['price'].mean():.2f} Cr")
c3.metric("Median Price", f"₹{filtered_df['price'].median():.2f} Cr")
c4.metric("Average Area", f"{filtered_df['area'].mean():,.0f} sq. ft.")

st.divider()

# Region insights
st.subheader("1. Region-wise Insights")

region_summary = (
    filtered_df.groupby("category")["price"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .rename(columns={
        "category": "Region",
        "mean": "Average Price (Cr)",
        "median": "Median Price (Cr)",
        "count": "Properties"
    })
    .sort_values("Median Price (Cr)", ascending=False)
)

st.dataframe(region_summary.round(2), use_container_width=True, hide_index=True)

fig = px.bar(
    region_summary,
    x="Region",
    y="Median Price (Cr)",
    title="Median Property Price by Region",
    hover_data=["Properties"]
)
st.plotly_chart(fig, use_container_width=True)

if not region_summary.empty:
    highest = region_summary.iloc[0]
    lowest = region_summary.iloc[-1]
    st.write(
        f"**Highest median price:** {highest['Region']} "
        f"(₹{highest['Median Price (Cr)']:.2f} Cr)."
    )
    st.write(
        f"**Lowest median price:** {lowest['Region']} "
        f"(₹{lowest['Median Price (Cr)']:.2f} Cr)."
    )

# Price ranges
st.subheader("2. Property Price Segments")

def price_segment(price):
    if price < 1:
        return "Below ₹1 Cr"
    elif price < 3:
        return "₹1–3 Cr"
    elif price < 5:
        return "₹3–5 Cr"
    return "₹5 Cr and above"

filtered_df["price_segment"] = filtered_df["price"].apply(price_segment)

segment_summary = (
    filtered_df["price_segment"]
    .value_counts()
    .reindex(
        ["Below ₹1 Cr", "₹1–3 Cr", "₹3–5 Cr", "₹5 Cr and above"],
        fill_value=0
    )
    .rename_axis("Price Segment")
    .reset_index(name="Properties")
)

fig = px.bar(
    segment_summary,
    x="Price Segment",
    y="Properties",
    title="Number of Properties in Each Price Segment"
)
st.plotly_chart(fig, use_container_width=True)

largest_segment = segment_summary.loc[segment_summary["Properties"].idxmax()]
st.write(
    f"**Most common price segment:** {largest_segment['Price Segment']} "
    f"({int(largest_segment['Properties']):,} properties)."
)

# Area insights
st.subheader("3. Area and Price Insights")

fig = px.scatter(
    filtered_df,
    x="area",
    y="price",
    color="category",
    hover_data=["bedrooms", "bathrooms", "type_of_building"],
    labels={"area": "Area (sq. ft.)", "price": "Price (Crores)"},
    title="Does Larger Area Generally Mean a Higher Price?"
)
st.plotly_chart(fig, use_container_width=True)

if filtered_df["area"].nunique() > 1 and filtered_df["price"].nunique() > 1:
    correlation = filtered_df["area"].corr(filtered_df["price"])
    st.write(f"**Area-price correlation:** {correlation:.2f}.")
    if correlation >= 0.6:
        st.info("In this dataset, area and price show a strong positive relationship.")
    elif correlation >= 0.3:
        st.info("In this dataset, area and price show a moderate positive relationship.")
    elif correlation > 0:
        st.info("In this dataset, area and price show a weak positive relationship.")
    else:
        st.info("In this dataset, area and price do not show a positive relationship.")

# Furnishing insights
st.subheader("4. Furnishing Insights")

furnishing_summary = (
    filtered_df.groupby("furnishing_label")["price"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .rename(columns={
        "furnishing_label": "Furnishing",
        "mean": "Average Price (Cr)",
        "median": "Median Price (Cr)",
        "count": "Properties"
    })
    .sort_values("Average Price (Cr)", ascending=False)
)

st.dataframe(
    furnishing_summary.round(2),
    use_container_width=True,
    hide_index=True
)

# Building type insights
st.subheader("5. Building Type Insights")

building_summary = (
    filtered_df.groupby("type_of_building")["price"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .rename(columns={
        "type_of_building": "Building Type",
        "mean": "Average Price (Cr)",
        "median": "Median Price (Cr)",
        "count": "Properties"
    })
    .sort_values("Median Price (Cr)", ascending=False)
)

fig = px.bar(
    building_summary,
    x="Building Type",
    y="Median Price (Cr)",
    title="Median Price by Building Type",
    hover_data=["Properties"]
)
st.plotly_chart(fig, use_container_width=True)

# Data quality
st.subheader("6. Data Quality")

q1, q2, q3 = st.columns(3)
q1.metric("Duplicate Rows", f"{filtered_df.duplicated().sum():,}")
q2.metric("Missing Values", f"{filtered_df.isna().sum().sum():,}")
q3.metric("Regions Covered", f"{filtered_df['category'].nunique()}")

with st.expander("View missing values by column"):
    missing_values = filtered_df.isna().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, use_container_width=True, hide_index=True)

