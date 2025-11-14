import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("Return Prediction Demo")

I_SIGNAL_CANDLE = 20

# Load sample windows
sample_windows = []
for i in range(5):
    df = pd.read_parquet(f"sample_data/w_{i}.parquet")
    sample_windows.append(df)

# Load model
models = {
    "Random Forest": joblib.load("models/rf.pkl"),
    "Extra Trees": joblib.load("models/ert.pkl")  # corrected, assuming ert.pkl exists
}

# Load sample data
X_samples = np.load("sample_data/X_flat_test.npy")
y_samples = np.load("sample_data/y_binary_test.npy")

# Let user pick a sample, with a placeholder first
options = ["Pick a sample window..."] + list(range(1, len(X_samples) + 1))
selected_option = st.selectbox("", options)

# Only run if a real sample is selected
if isinstance(selected_option, int):
    selected_idx = selected_option - 1  # 0-based index

    w = sample_windows[selected_idx]
    df = w[:21]
    ticker = df['ticker'].iloc[0]
    date = df['t'].iloc[0].date()
    signal_time = df['t'].iloc[I_SIGNAL_CANDLE].time()
    signal_time_str = signal_time.strftime("%H:%M")

    st.markdown(f"""
    **Sample Window Info**
    - **Ticker:** {ticker}
    - **Date:** {date}
    - **Time of signal candle:** {signal_time_str}
    """)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df['t'].dt.strftime('%H:%M'),
            open=df['o'],
            high=df['h'],
            low=df['l'],
            close=df['c'],
            name="Candles"
        )
    ])

    fig.update_xaxes(tickmode="linear", dtick=2)

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_title_font={"size": 18},
        yaxis_title_font={"size": 18},
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Select model
    options_models = ["Select a model..."] + list(models.keys())
    selected_model_name = st.selectbox("", options_models)
    selected_model = models[selected_model_name]
    st.write(selected_model)
    if selected_model_name != "Select a model...":
        selected_model = models[selected_model_name]
        st.write(f"Selected model: {selected_model_name}")

        sample_input = X_samples[selected_idx]
        sample_input = sample_input.reshape(1, -1)  # shape (1, n_features)
        true_label = y_samples[selected_idx]
        prediction = selected_model.predict_proba(sample_input)

        st.write("True label:", true_label)
        st.write("Prediction:", prediction[0, 1])


    else:
        selected_model = None
        st.write("No model selected.")
