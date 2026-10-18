from pathlib import Path

import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Property Recommendations",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Mumbai Property Recommendations")
st.write("Find properties that match your budget and preferences.")

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

# Prepare data
for col in ["price", "area", "bedrooms", "bathrooms"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price", "area"])
df = df[(df["price"] >= 0) & (df["area"] > 0)]

if df.empty:
    st.error("No valid properties found in the dataset.")
    st.stop()

# Convert furnishing codes to labels
furnishing_map = {
    "0": "Unfurnished", "0.0": "Unfurnished",
    "1": "Semi-Furnished", "1.0": "Semi-Furnished",
    "2": "Fully Furnished", "2.0": "Fully Furnished"
}
df["furnishing_label"] = (
    df["furnished_status"].astype(str).str.strip().replace(furnishing_map)
)

# Recommendation form
st.subheader("Choose your preferences")

with st.form("recommendation_form"):
    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input(
            "Maximum budget (Crores)",
            min_value=0.1,
            value=2.0,
            step=0.1
        )

        min_area = st.number_input(
            "Minimum area (sq. ft.)",
            min_value=0,
            value=500,
            step=50
        )

        bedrooms = st.selectbox(
            "Bedrooms",
            ["Any"] + sorted(df["bedrooms"].dropna().unique().tolist())
        )

        region = st.selectbox(
            "Preferred region",
            ["Any"] + sorted(df["category"].dropna().unique().tolist())
        )

    with col2:
        bathrooms = st.selectbox(
            "Bathrooms",
            ["Any"] + sorted(df["bathrooms"].dropna().unique().tolist())
        )

        furnishing = st.selectbox(
            "Furnishing",
            ["Any"] + sorted(df["furnishing_label"].dropna().unique().tolist())
        )

        building_type = st.selectbox(
            "Building type",
            ["Any"] + sorted(df["type_of_building"].dropna().unique().tolist())
        )

        status = st.selectbox(
            "Property status",
            ["Any"] + sorted(df["status"].dropna().unique().tolist())
        )

    submitted = st.form_submit_button("Find Properties")

if submitted:
    recommendations = df[
        (df["price"] <= budget)
        & (df["area"] >= min_area)
    ].copy()

    if bedrooms != "Any":
        recommendations = recommendations[recommendations["bedrooms"] == bedrooms]

    if bathrooms != "Any":
        recommendations = recommendations[recommendations["bathrooms"] == bathrooms]

    if region != "Any":
        recommendations = recommendations[recommendations["category"] == region]

    if furnishing != "Any":
        recommendations = recommendations[recommendations["furnishing_label"] == furnishing]

    if building_type != "Any":
        recommendations = recommendations[recommendations["type_of_building"] == building_type]

    if status != "Any":
        recommendations = recommendations[recommendations["status"] == status]

    recommendations = recommendations.sort_values(
        by=["price", "area"],
        ascending=[True, False]
    )

    st.divider()
    st.subheader("Recommended Properties")

    if recommendations.empty:
        st.warning("No properties match all your preferences. Try increasing your budget or reducing the minimum area.")
    else:
        st.success(f"Found {len(recommendations):,} matching properties.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Matching Properties", f"{len(recommendations):,}")
        c2.metric("Average Price", f"₹{recommendations['price'].mean():.2f} Cr")
        c3.metric("Average Area", f"{recommendations['area'].mean():,.0f} sq. ft.")

        display_cols = [
            "price", "area", "bedrooms", "bathrooms", "category",
            "furnishing_label", "type_of_building", "status", "neworold"
        ]

        st.dataframe(
            recommendations[display_cols].rename(columns={
                "price": "Price (Cr)",
                "area": "Area (sq. ft.)",
                "bedrooms": "Bedrooms",
                "bathrooms": "Bathrooms",
                "category": "Region",
                "furnishing_label": "Furnishing",
                "type_of_building": "Building Type",
                "status": "Status",
                "neworold": "New / Resale"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Recommendations (CSV)",
            data=recommendations[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="recommended_properties.csv",
            mime="text/csv"
        )
else:
    st.info("Select your preferences and click 'Find Properties' to see matching properties.")

