import streamlit as st

st.set_page_config(page_title="House Price Prediction")

st.title("🏠 House Price Prediction & Valuation Tool")

st.sidebar.header("Enter House Details")

quality = st.sidebar.slider("Overall Quality", 1, 10, 5)
area = st.sidebar.number_input("Living Area", 500, 5000, 1500)
garage = st.sidebar.slider("Garage Cars", 0, 5, 2)

if st.button("Predict Price"):

    price = quality * 50000 + area * 100 + garage * 10000

    st.success(f"Estimated House Price: ${price:,.2f}")

st.write("Interactive Streamlit Deployment Successful ✅")
