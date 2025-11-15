import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

I_SIGNAL_CANDLE = 20
I_BUY_CANDLE = 21
I_SELL_CANDLE = 25

# Load data


# Original test windows set
with open("windows_test.pkl", "rb") as f:
    windows_test = pickle.load(f)

# Trade metrics for (model, threshold) pairs
with open("metrics_to_streamlit.pkl", "rb") as f:
    metrics_to_streamlit = pickle.load(f)


# Get baseline metrics as separate object
for i, m in enumerate(metrics_to_streamlit):
    if m.name == 'baseline':
        baseline_metrics = m
        del(metrics_to_streamlit[i])

options_thresholds_probs = []
options_thresholds_rtns = []
options_models = set()
for m in metrics_to_streamlit:
    name = m.name
    thr = m.threshold
    if name[-4:] = '_reg':
        options_thresholds_rtns.append(thr)
    else:
        options_thresholds_probs.append(thr)
    options_models.add(name)

st.title("Trade selection using machine learning and candlestick patterns")

selected_option = st.selectbox("", options_thresholds_rtns)
