import streamlit as st

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction & Valuation Tool")

st.write("Predict estimated house prices using house features.")

st.sidebar.header("Enter House Details")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Ground Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.sidebar.slider("Garage Capacity", 0, 5, 2)
total_bsmt_sf = st.sidebar.number_input("Basement Area", 0, 3000, 800)
year_built = st.sidebar.slider("Year Built", 1900, 2025, 2005)

if st.button("Predict House Price"):

    estimated_price = (
        overall_qual * 50000
        + gr_liv_area * 120
        + garage_cars * 10000
        + total_bsmt_sf * 35
        + (year_built - 2000) * 1500
    )

    st.success(f"🏡 Estimated House Price: ${estimated_price:,.2f}")

    st.subheader("📊 Input Summary")

    st.write(f"Overall Quality: {overall_qual}")
    st.write(f"Living Area: {gr_liv_area} sq ft")
    st.write(f"Garage Capacity: {garage_cars}")
    st.write(f"Basement Area: {total_bsmt_sf} sq ft")
    st.write(f"Year Built: {year_built}")

st.markdown("---")
st.caption("Machine Learning House Price Prediction Project")
