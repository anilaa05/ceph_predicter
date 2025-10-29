import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(page_title="Ceph Scrubbing Forecast", page_icon="🔮", layout="centered")

# ---- Custom Background & Style ----
page_bg = """
<style>
/* Main page background */
[data-testid="stAppViewContainer"] {
    background-color: #53b4e6;
    color: white;
}

/* Top header + sidebar */
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1);
}

/* Headings and text */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}

/* File uploader styling */
section[data-testid="stFileUploader"] {
    background-color: white !important;
    border-radius: 12px;
    padding: 20px;
    border: 3px dashed #53b4e6 !important;
}

section[data-testid="stFileUploader"] label {
    color: #53b4e6 !important;
    font-weight: bold;
}

div[data-testid="stFileUploaderDropzone"] div {
    color: #53b4e6 !important;
}

/* DataFrame box */
.stDataFrame {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton>button {
    background-color: white;
    color: #0077c8;
    border-radius: 8px;
    height: 2.5em;
    width: 100%;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #e0f7ff;
    color: #005b99;
}

/* Slider label */
div[data-testid="stSlider"] > label {
    color: white !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1 style='text-align:center; color:white;'>🔮 Ceph Scrubbing Count Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload your Ceph data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # ---- Preprocess ----
    df['ds'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df['y'] = df['scrubbing']
    df = df.dropna(subset=['ds', 'y'])
    
    st.success("✅ Data loaded successfully!")
    st.write("### 📊 Preview of your data:")
    st.dataframe(df.head())

    # ---- Train Prophet ----
    model = Prophet()
    model.fit(df)
    
    # ---- Future Forecast ----
    periods = st.slider("⏱️ Predict how many future days?", 1, 30, 7)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    # ---- Show forecast ----
    st.subheader("📈 Forecast Results")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("📉 Trend Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
    
    st.success("🎯 Prediction complete!")

else:
    st.info("👆 Upload a Ceph dataset (with timestamp & scrubbing columns) to begin.")

