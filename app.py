import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelTradesMetrics:
    name: str
    threshold: float
    n_trades: int
    n_correct: int
    mean_return: float
    hitrate: float
    sharpe: float
    indices: List[int]
    returns: List[float]
    perm_pval_mean: Optional[float] = None
    perm_pval_hitrate: Optional[float] = None
    perm_pval_sharpe: Optional[float] = None
    boot_p_val_mean: Optional[float] = None
    
# ----------------------------
# Load pre-saved data
# ----------------------------

def load_data():
    with open("data/metrics_to_streamlit.pkl", "rb") as f:
        metrics = pickle.load(f)
    with open("data/windows_test.pkl", "rb") as f:
        windows = pickle.load(f)
    return metrics, windows

metrics_to_streamlit, windows_test = load_data()

# ----------------------------d
# Sidebar: Model and Threshold Selection
# ----------------------------
st.sidebar.header("Select Model & Threshold")

# List of model names
model_names = sorted(list(set([m.name for m in metrics_to_streamlit])))
selected_model = st.sidebar.selectbox("Model", model_names)

# Get thresholds available for this model
thresholds = [m.threshold for m in metrics_to_streamlit if m.name == selected_model]
thresholds = sorted(thresholds)
selected_threshold = st.sidebar.selectbox("Threshold", thresholds)

# ----------------------------
# Fetch corresponding metrics object
# ----------------------------
selected_metrics = next(
    m for m in metrics_to_streamlit
    if m.name == selected_model and m.threshold == selected_threshold
)

# ----------------------------
# Compute DataFrame for selected trades
# ----------------------------
selected_windows = [windows_test[i] for i in selected_metrics.indices]
trades_df = pd.concat(selected_windows, ignore_index=True)
trades_df["return"] = selected_metrics.returns
trades_df["hit"] = [1 if r > 0 else 0 for r in selected_metrics.returns]

# ----------------------------
# Display KPIs
# ----------------------------
st.subheader("Trade Metrics")
st.metric("Number of trades", selected_metrics.n_trades)
st.metric("Hit rate", f"{selected_metrics.hitrate:.2f}" if selected_metrics.hitrate is not None else "N/A")
st.metric("Mean return", f"{selected_metrics.mean_return:.4f}" if selected_metrics.mean_return is not None else "N/A")
st.metric("Sharpe ratio", f"{selected_metrics.sharpe:.2f}" if selected_metrics.sharpe is not None else "N/A")

# ----------------------------
# Show all trades in scrollable table
# ----------------------------
st.subheader("Selected Trades")
with st.expander(f"Show all {selected_metrics.n_trades} trades"):
    st.dataframe(trades_df)

# ----------------------------
# Plot individual trade candlestick
# ----------------------------
st.subheader("Individual Trade Chart")
trade_index = st.slider("Select trade index", 0, max(0, selected_metrics.n_trades - 1), 0)

if selected_metrics.n_trades > 0:
    trade_window = windows_test[selected_metrics.indices[trade_index]]

    fig = go.Figure(data=[go.Candlestick(
        x=trade_window['t'],
        open=trade_window['open'],
        high=trade_window['high'],
        low=trade_window['low'],
        close=trade_window['close']
    )])
    st.plotly_chart(fig)
else:
    st.write("No trades selected for this model/threshold.")
