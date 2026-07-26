
import streamlit as st
import pandas as pd
import requests

# Backend API URL
BACKEND_URL = "http://backend:7860"

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Product Store Sales Prediction")
st.markdown(
    "Predict the **sales revenue** of a product in a particular store using the trained Machine Learning model."
)

# -----------------------------------------------------
# Online Prediction
# -----------------------------------------------------

st.header("📈 Single Prediction")

col1, col2 = st.columns(2)

with col1:

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.5
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.000,
        max_value=1.000,
        value=0.05,
        format="%.3f"
    )

    product_type = st.selectbox(
        "Product Type",
        [
            "Baking Goods",
            "Breads",
            "Breakfast",
            "Canned",
            "Dairy",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Hard Drinks",
            "Health and Hygiene",
            "Household",
            "Meat",
            "Others",
            "Seafood",
            "Snack Foods",
            "Soft Drinks",
            "Starchy Foods"
        ]
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0
    )

with col2:

    store_id = st.selectbox(
        "Store ID",
        [
            "OUT001",
            "OUT002",
            "OUT003",
            "OUT004"
        ]
    )

    store_establishment_year = st.selectbox(
        "Store Establishment Year",
        [
            1987,
            1998,
            1999,
            2009
        ]
    )

    store_size = st.selectbox(
        "Store Size",
        [
            "Small",
            "Medium",
            "High"
        ]
    )

    store_location_city_type = st.selectbox(
        "Store City Tier",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Food Mart",
            "Supermarket Type1",
            "Supermarket Type2"
        ]
    )

# -----------------------------------------------------
# Prepare Input
# -----------------------------------------------------

input_data = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}

# -----------------------------------------------------
# Prediction Button
# -----------------------------------------------------

if st.button("Predict Sales", use_container_width=True):

    response = requests.post(
        f"{BACKEND_URL}/v1/predict",
        json=input_data
    )

    if response.status_code == 200:

        prediction = response.json()["Predicted_Product_Store_Sales_Total"]

        st.success(
            f"### Predicted Product Store Sales: ₹ {prediction:,.2f}"
        )

    else:
        st.error("Unable to connect to the prediction API.")

# -----------------------------------------------------
# Batch Prediction
# -----------------------------------------------------

st.divider()

st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    st.write("Preview of Uploaded File")

    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    if st.button("Predict Batch Sales", use_container_width=True):

        uploaded_file.seek(0)

        response = requests.post(
            f"{BACKEND_URL}/v1/predict_batch",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:

            predictions = response.json()["Predictions"]

            df["Predicted_Product_Store_Sales_Total"] = predictions

            st.success("Batch Prediction Completed Successfully!")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="product_store_sales_predictions.csv",
                mime="text/csv"
            )

        else:

            st.error("Unable to connect to the prediction API.")
