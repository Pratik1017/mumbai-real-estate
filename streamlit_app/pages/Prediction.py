
import streamlit as st
import requests

st.set_page_config(
    page_title="Property Price Prediction",
    page_icon="🏠",
    layout="wide"
)

API_URL = "http://api:8000/predict"

st.title("🏠 Mumbai Property Price Prediction")
if st.button("🔄 Reload Page"):
    st.rerun()
st.write("Enter property details to estimate its market price.")

st.divider()

with st.form("prediction_form"):

    st.subheader("📋 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=1.0,
            value=1000.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        status = st.selectbox(
            "Property Status",
            ["Ready to Move", "Under Construction"]
        )

    with col2:
        neworold = st.selectbox(
            "Property Type",
            ["New Property", "Resale"]
        )

        furnished_label = st.selectbox(
            "Furnishing Status",
            [
                "Unfurnished",
                "Semi-Furnished",
                "Fully Furnished"
            ]
        )

        type_of_building = st.selectbox(
            "Building Type",
            ["Flat", "Individual House"]
        )

        category = st.selectbox(
            "Mumbai Region",
            [
                "North Mumbai",
                "South Mumbai",
                "East Mumbai",
                "West Mumbai"
            ]
        )

    predict_button = st.form_submit_button(
        "Predict Property Price",
        type="primary",
        use_container_width=True
    )


if predict_button:

    # Convert furnishing label to model's numeric encoding
    furnishing_mapping = {
        "Unfurnished": 0,
        "Semi-Furnished": 1,
        "Fully Furnished": 2
    }

    user_input = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "status": status,
        "neworold": neworold,
        "furnished_status": furnishing_mapping[furnished_label],
        "type_of_building": type_of_building,
        "category": category
    }

    with st.spinner("Predicting property price..."):

        try:
            response = requests.post(
                API_URL,
                json=user_input,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            predicted_price = float(result["predicted_price"])

            st.divider()
            st.subheader("💰 Prediction Result")

            st.metric(
                label="Estimated Property Price",
                value=f"₹ {predicted_price:.2f} Crore"
            )

            st.success("Property price predicted successfully!")

            with st.expander("View Submitted Property Details"):
                st.json(user_input)

            st.caption(
                "Note: This is an ML-generated estimate, "
                "not a guaranteed market valuation."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Please start your backend server."
            )

        except requests.exceptions.Timeout:
            st.error("Prediction request timed out. Please try again.")

        except requests.exceptions.HTTPError:
            st.error(
                f"API error ({response.status_code}): "
                f"{response.text}"
            )

        except (KeyError, ValueError, TypeError):
            st.error("Invalid prediction response received from the API.")
