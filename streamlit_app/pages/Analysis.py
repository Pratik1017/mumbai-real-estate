from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Mumbai Real Estate Analysis", page_icon="📊", layout="wide")
st.title("📊 Mumbai Real Estate Analysis")
st.write("Explore property prices, regions, and property features.")

# Load dataset
DATA_PATH = "data/base-file/analysis.csv"  # Change this path if needed

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# Check required columns
required_cols = [
    "price", "area", "bedrooms", "bathrooms", "category",
    "furnished_status", "type_of_building", "status", "neworold"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# Clean basic data
for col in ["price", "area", "bedrooms", "bathrooms"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price", "area"])
df = df[(df["price"] >= 0) & (df["area"] > 0)]

if df.empty:
    st.error("No valid property records found.")
    st.stop()

# Convert furnishing codes to labels (check this mapping against your data)
furnishing_map = {
    "0": "Unfurnished", "0.0": "Unfurnished",
    "1": "Semi-Furnished", "1.0": "Semi-Furnished",
    "2": "Fully Furnished", "2.0": "Fully Furnished"
}
df["furnishing_label"] = (
    df["furnished_status"].astype(str).str.strip().replace(furnishing_map)
)

# Sidebar filters
st.sidebar.header("Filters")

regions = sorted(df["category"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

min_price = float(df["price"].min())
max_price = float(df["price"].max())

if min_price < max_price:
    selected_price = st.sidebar.slider(
        "Price range (Crores)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )
else:
    selected_price = (min_price, max_price)

furnishing_options = sorted(df["furnishing_label"].dropna().unique().tolist())
selected_furnishing = st.sidebar.multiselect(
    "Furnishing", furnishing_options, default=furnishing_options
)

building_options = sorted(df["type_of_building"].dropna().unique().tolist())
selected_buildings = st.sidebar.multiselect(
    "Building type", building_options, default=building_options
)

# Apply filters
filtered_df = df[
    df["category"].isin(selected_regions)
    & df["price"].between(selected_price[0], selected_price[1])
    & df["furnishing_label"].isin(selected_furnishing)
    & df["type_of_building"].isin(selected_buildings)
].copy()

if filtered_df.empty:
    st.warning("No properties match the selected filters.")
    st.stop()

# Main metrics
st.subheader("Market Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Properties", f"{len(filtered_df):,}")
c2.metric("Average price", f"₹{filtered_df['price'].mean():.2f} Cr")
c3.metric("Median price", f"₹{filtered_df['price'].median():.2f} Cr")
c4.metric("Average area", f"{filtered_df['area'].mean():,.0f} sq. ft.")

st.caption(f"Showing {len(filtered_df):,} of {len(df):,} properties.")
st.divider()

# 1. Price distribution
st.subheader("1. Price Distribution")
fig = px.histogram(
    filtered_df, x="price", color="category", nbins=30,
    title="Property prices by region",
    labels={"price": "Price (Crores)", "category": "Region"}
)
st.plotly_chart(fig, use_container_width=True)

# 2. Region comparison
st.subheader("2. Region-wise Price")
region_stats = (
    filtered_df.groupby("category")["price"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .rename(columns={
        "category": "Region", "mean": "Average Price",
        "median": "Median Price", "count": "Properties"
    })
)

fig = px.bar(
    region_stats, x="Region", y=["Average Price", "Median Price"],
    barmode="group", title="Average vs Median Price"
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(region_stats.round(2), use_container_width=True, hide_index=True)

# 3. Area vs price
st.subheader("3. Area vs Price")
fig = px.scatter(
    filtered_df, x="area", y="price", color="category",
    hover_data=["bedrooms", "bathrooms"],
    labels={"area": "Area (sq. ft.)", "price": "Price (Crores)"}
)
st.plotly_chart(fig, use_container_width=True)

# 4. Furnishing and building type
left, right = st.columns(2)

with left:
    st.subheader("4. Price by Furnishing")
    furnishing_stats = (
        filtered_df.groupby("furnishing_label")["price"]
        .mean().reset_index()
    )
    fig = px.bar(
        furnishing_stats, x="furnishing_label", y="price",
        labels={"furnishing_label": "Furnishing", "price": "Average Price (Crores)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("5. Price by Building Type")
    fig = px.box(
        filtered_df, x="type_of_building", y="price",
        labels={"type_of_building": "Building Type", "price": "Price (Crores)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# 6. Correlation
st.subheader("6. Numerical Correlation")
numeric_cols = ["price", "area", "bedrooms", "bathrooms"]
corr = filtered_df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Correlation between numeric columns")
st.plotly_chart(fig, use_container_width=True)

# Key findings
st.subheader("Key Findings")
region_stats = region_stats.dropna(subset=["Median Price"])
if not region_stats.empty:
    highest = region_stats.loc[region_stats["Median Price"].idxmax()]
    lowest = region_stats.loc[region_stats["Median Price"].idxmin()]
    st.write(f"**Highest median price:** {highest['Region']} — ₹{highest['Median Price']:.2f} Cr")
    st.write(f"**Lowest median price:** {lowest['Region']} — ₹{lowest['Median Price']:.2f} Cr")

st.write(f"**Median area:** {filtered_df['area'].median():,.0f} sq. ft.")