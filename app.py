import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title("Return Prediction Demo")

# Load model
rf_model = joblib.load("models/rf.pkl")

# Load sample data
X_samples = np.load("sample_data/X_flat_test.npy")
y_samples = np.load("sample_data/y_binary_test.npy")

# Let user pick a sample
selected_idx = -1 + st.selectbox("Pick a sample window", list(range(1, len(X_samples) + 1)))
sample_input = X_samples[selected_idx]
true_label = y_samples[selected_idx]

# Load sample windows
sample_windows = []
for i in range(5):
    df = pd.read_parquet(f"sample_data/w_{i}.parquet")
    sample_windows.append(df)

w = sample_windows[selected_idx]
df = w[:21]
ticker = df['ticker'].iloc[0]
date = df['t'].iloc[0].date()

import plotly.graph_objects as go


# Plotly candlestick figure
fig = go.Figure(data=[
    go.Candlestick(
        x=df['t'],
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        name="Candles"
    )
])

fig.update_layout(
    title={
        "text": f"{ticker}, {date}",
        "font": {"size": 28}   # adjust title size here
    },
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_title_font={"size": 20},
    yaxis_title_font={"size": 20},
    xaxis_rangeslider_visible=False
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write("Ticker:", ticker)
st.write("Date:", date)
st.write("Time of first candle:", w['t'].iloc[0])
st.write("Time of signal candle:", w['t'].iloc[21])

sample_input = X_samples[selected_idx]
sample_input = X_samples[selected_idx].reshape(1, -1) # shape (1, n_features) for tree models

st.write("True label:", true_label)
# Make prediction
prediction = rf_model.predict_proba(sample_input)
st.write("Prediction:", prediction[0, 1])
