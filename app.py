import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model

# ======================
# Load Data & Model
# ======================
df = pd.read_excel(".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx")
model = load_model(".venv/volatility_secure_model.h5")

# ======================
# Data Security (Hashing)
# ======================
def hash_value(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

df["Hashed_ID"] = df["State"].apply(hash_value)

# ======================
# Dashboard Layout
# ======================
st.title("🎯 US Toy Manufacturers Dashboard")

# Sidebar filters
state = st.sidebar.selectbox("Select State", df["State"].unique())
year = st.sidebar.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()))

# Filtered Data
state_data = df[df["State"] == state]

# ======================
# Graph 1: Bar Graph (Top 10 states)
# ======================
st.subheader("Top 10 States by Manufacturers (Total)")
top10 = df.groupby("State")["Number of Manufacturers"].sum().nlargest(10).reset_index()
fig1 = px.bar(top10, x="State", y="Number of Manufacturers", color="Number of Manufacturers",
              title="Top 10 States Demand")
st.plotly_chart(fig1)

# ======================
# Graph 2: Pie Chart
# ======================
st.subheader("Share of Manufacturers by Top 10 States")
fig2 = px.pie(top10, names="State", values="Number of Manufacturers", title="Top 10 State Share")
st.plotly_chart(fig2)

# ======================
# Graph 3: Area Chart
# ======================
st.subheader(f"Demand Trend for {state}")
fig3 = px.area(state_data, x="Year", y="Number of Manufacturers", title=f"Yearly Demand in {state}")
st.plotly_chart(fig3)

# ======================
# Prediction using NN
# ======================
st.subheader("📈 Neural Network Prediction")
recent_data = state_data["Number of Manufacturers"].values[-5:].reshape(1, 5, 1)
pred = model.predict(recent_data)[0][0]
st.write(f"Predicted demand volatility score for {state} = {pred:.2f}")

# ======================
# Display Hashed IDs
# ======================
st.subheader("🔐 Secure Data (Hashed IDs)")
st.dataframe(df[["State", "Hashed_ID"]].drop_duplicates())
