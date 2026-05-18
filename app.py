# =====================================================
# HOUSE PRICE PREDICTION STREAMLIT APP
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("final_house_price_model.pkl")

# =====================================================
# LOAD DATASET SAMPLE
# =====================================================

sample_data = pd.read_csv("AmesHousing.csv")

# =====================================================
# TITLE
# =====================================================

st.title("🏠 House Price Prediction & Valuation Tool")

st.write(
    "Predict house prices using Machine Learning & XGBoost"
)

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("🏡 Enter Property Details")

overall_qual = st.sidebar.slider(
    "Overall Quality",
    1,
    10,
    5
)

gr_liv_area = st.sidebar.number_input(
    "Ground Living Area",
    500,
    5000,
    1500
)

garage_cars = st.sidebar.slider(
    "Garage Cars",
    0,
    5,
    2
)

garage_area = st.sidebar.number_input(
    "Garage Area",
    0,
    2000,
    500
)

total_bsmt_sf = st.sidebar.number_input(
    "Total Basement Area",
    0,
    3000,
    800
)

year_built = st.sidebar.slider(
    "Year Built",
    1900,
    2025,
    2000
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    0,
    5,
    2
)

totrms_abvgrd = st.sidebar.slider(
    "Total Rooms Above Ground",
    1,
    15,
    7
)

# =====================================================
# PREPARE INPUT DATA
# =====================================================

input_data = sample_data.iloc[[0]].copy()

input_data["Overall Qual"] = overall_qual

input_data["Gr Liv Area"] = gr_liv_area

input_data["Garage Cars"] = garage_cars

input_data["Garage Area"] = garage_area

input_data["Total Bsmt SF"] = total_bsmt_sf

input_data["Year Built"] = year_built

input_data["Full Bath"] = full_bath

input_data["TotRms AbvGrd"] = totrms_abvgrd

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict House Price"):

    prediction = model.predict(input_data)

    prediction_price = np.expm1(prediction[0])

    st.success(
        f"🏠 Estimated House Price: ${prediction_price:,.2f}"
    )

    st.subheader("📋 Property Details")

    st.write(input_data)

# =====================================================
# MODEL INFO
# =====================================================

st.markdown("---")

st.header("📊 Model Information")

st.write("""

This application predicts house prices using:

- XGBoost Regression
- Feature Engineering
- Data Preprocessing Pipelines
- Machine Learning Regression Techniques

""")
