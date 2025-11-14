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
selected_idx = 0 - st.selectbox("Pick a sample window", list(range(1, len(X_samples) + 1)))
sample_input = X_samples[selected_idx]
true_label = y_samples[selected_idx]

# Load sample windows
sample_windows = []
for i in range(5):
    df = pd.read_parquet(f"sample_data/w_{i}.parquet")
    sample_windows.append(df)


st.write("Ticker:", sample_windows[selected_idx]['ticker'].iloc[0])
st.write("Date:", sample_windows[selected_idx]['t'].iloc[0].date())
st.write("Time of first candle:", sample_windows[selected_idx]['t'].iloc[0])
st.write("Time of signal candle:", sample_windows[selected_idx]['t'].iloc[21])

sample_input = X_samples[selected_idx]
sample_input = X_samples[selected_idx].reshape(1, -1) # shape (1, n_features) for tree models

# Show input and label
st.write("Selected sample features:", sample_input)
st.write("True label:", true_label)

# Make prediction
prediction = rf_model.predict_proba(sample_input)
st.write("Prediction:", prediction[0, 1])
