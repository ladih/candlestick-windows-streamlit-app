import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from load_models import models  # Import the dictionary from load_models.py
from sklearn.ensemble import RandomForestClassifier

st.title("Return Prediction Demo")

I_SIGNAL_CANDLE = 20
I_BUY_CANDLE = 21
I_SELL_CANDLE = 25

# Load sample windows
sample_windows = []
for i in range(5):
    df = pd.read_parquet(f"sample_data/w_{i}.parquet")
    sample_windows.append(df)

# Load model


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
    signal_candle_perc = (w['c'].iloc[I_SIGNAL_CANDLE] - w['o'].iloc[I_SIGNAL_CANDLE]) / w['o'].iloc[I_SIGNAL_CANDLE]
    rtn = (w['c'].iloc[I_SELL_CANDLE] - w['o'].iloc[I_BUY_CANDLE]) / w['o'].iloc[I_BUY_CANDLE]
    if rtn > 0:
        rtn_direction = 'positive'
    else:
        rtn_direction = 'negative'

    st.markdown(f"""
    **Sample Window Info**
    - **Ticker:** {ticker}
    - **Date:** {date}
    - **Time of signal candle:** {signal_time_str}
    - **Signal candle percentage:** {signal_candle_perc * 100:.1f} %
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
    title={
        'text': "Chart up to signal candle",
        'x': 0.5,  # Centers the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {"size": 20}
    },
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_title_font={"size": 16},
    yaxis_title_font={"size": 16},
    xaxis_rangeslider_visible=False
)
    st.plotly_chart(fig, use_container_width=True)

    # Select model
    options_models = ["Select a model..."] + list(models.keys())
    selected_model_name = st.selectbox("", options_models)
    if selected_model_name != "Select a model...":
        selected_model = models[selected_model_name]
        st.write(f"Selected model: {selected_model_name}")
        if selected_model_name.split()[-1] == 'regression':
            model_type = 'regression'
        else:
            model_type = 'classification'
        sample_input = X_samples[selected_idx]
        sample_input = sample_input.reshape(1, -1)  # shape (1, n_features)
        true_label = y_samples[selected_idx]
        prediction = selected_model.predict_proba(sample_input)

        if model_type == 'regression':
            st.write(f"{selected_model_name} predicts that the return is: {prediction[0, 1]:.2f}")
        else:
            st.write(f"{selected_model_name} predicts that the probability of positive return is: {prediction[0, 1]:.2f}")
            # Determine color based on rtn_direction
            color = "green" if rtn_direction.lower() == "positive" else "red"

            # Display the text with colored rtn_direction
            st.markdown(
                f'The actual return is: <span style="color:{color}; font-weight:bold;">{rtn_direction}</span> ({100 * rtn:.2f} %)',
                unsafe_allow_html=True
                st.write(f"The actual return is: {rtn_direction} ({100 * rtn:.2f} %)")
