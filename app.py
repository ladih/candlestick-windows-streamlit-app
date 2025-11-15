import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from model_trade_metric_class import ModelTradesMetrics

I_SIGNAL_CANDLE = 20
I_BUY_CANDLE = 21
I_SELL_CANDLE = 25

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
if selected_model == "baseline":
    thresholds = ["N/A"]
else:
    thresholds = [m.threshold for m in metrics_to_streamlit if m.name == selected_model]
thresholds = sorted(thresholds)
selected_threshold = st.sidebar.selectbox("Threshold", thresholds)

# ----------------------------
# Fetch corresponding metrics object
# ----------------------------

if selected_threshold == 'N/A':
    selected_metrics = next(m for m in metrics_to_streamlit if m.name == 'baseline')
else:
    selected_metrics = next(
        m for m in metrics_to_streamlit
        if m.name == selected_model and m.threshold == selected_threshold
    )
has_trades = selected_metrics.n_trades > 0
# ----------------------------
# Compute DataFrame for selected trades
# ----------------------------
selected_windows = [windows_test[i] for i in selected_metrics.indices]

sig_times = [windows_test[i]['t'].iloc[I_SIGNAL_CANDLE].strftime('%H:%M') for i in selected_metrics.indices]
trades_df = pd.DataFrame({
    "Date": [windows_test[i]['t'].iloc[0].date() for i in selected_metrics.indices],  # first candle date
    "Ticker": [windows_test[i]['ticker'].iloc[0] for i in selected_metrics.indices],
    "Signal candle time": sig_times,
    "Hit?": [1 if r > 0 else 0 for r in selected_metrics.returns],               # profitable or not
    "Return": selected_metrics.returns                                        # actual returns
})
trades_df.index.name = "Index"
# ----------------------------
# Display KPIs
# ----------------------------
thr = 'N/A' if selected_metrics.name == 'baseline' else selected_metrics.threshold
st.subheader(f"Metrics for trades selected by ({selected_metrics.name}, {thr})")
st.metric("Number of trades", selected_metrics.n_trades)
st.metric("Hit rate", f"{selected_metrics.hitrate:.2f}" if selected_metrics.hitrate is not None else "N/A")
st.metric("Mean return", f"{selected_metrics.mean_return:.4f}" if selected_metrics.mean_return is not None else "N/A")
st.metric("Sharpe ratio", f"{selected_metrics.sharpe:.2f}" if selected_metrics.sharpe is not None else "N/A")

# ----------------------------
# Show all trades in scrollable table
# ----------------------------

st.subheader("Selected Trades")

# Center all numeric columns
styled_df = trades_df.style.set_properties(
    **{'text-align': 'center'}, subset=trades_df.select_dtypes(include=['number']).columns
)
if has_trades:
    with st.expander(f"Show all {selected_metrics.n_trades} trades"):
        st.dataframe(styled_df)
else:
    st.write("No trades to show")

# ----------------------------
# Plot individual trade candlestick
# ----------------------------
st.subheader("Individual Trade Chart")
if has_trades:
    trade_index = st.slider("Select trade index", 0, max(0, selected_metrics.n_trades - 1), 0)

    trade_window = windows_test[selected_metrics.indices[trade_index]]
    ticker = trade_window['ticker'].iloc[0]
    date = trade_window['t'].iloc[0].date()
    time_signal = trade_window['t'][I_SIGNAL_CANDLE].strftime('%H:%M')
    rtn_signal = (trade_window['c'][I_SIGNAL_CANDLE] - trade_window['o'][I_SIGNAL_CANDLE]) / \
                    trade_window['o'][I_SIGNAL_CANDLE]

    st.markdown(f"""
    The chosen trade has:
    - **Ticker:** {ticker}
    - **Date:** {date}
    - **Signal candle time:** {time_signal}
    - **Signal candle percentage:** {100 * rtn_signal:.2f} %
    - **Return:** {100 * selected_metrics.returns[trade_index]:.2f} %
    """)


    fig = go.Figure(data=[go.Candlestick(
        x=trade_window['t'].dt.strftime('%H:%M'),
        open=trade_window['o'],
        high=trade_window['h'],
        low=trade_window['l'],
        close=trade_window['c']
    )])

    buy_x = trade_window['t'].dt.strftime('%H:%M')[I_BUY_CANDLE]
    buy_y = trade_window['o'][I_BUY_CANDLE]
    up_buy_candle = trade_window['c'][I_BUY_CANDLE] - trade_window['o'][I_BUY_CANDLE] > 0

    if up_buy_candle:
        dir_buy = 1
    else:
        dir_buy = -1

    # annotation (text above)
    fig.add_annotation(
        x=buy_x,      # arrow point (candle)
        y=buy_y,
        ax=0,         # shift arrow tail horizontally (0 = same x)
        ay=dir_buy * 70,        # positive = above, bigger number = longer tail
        text="BUY",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor="green",
        font=dict(size=14, color="green")
    )

    # SELL
    sell_x = trade_window['t'].dt.strftime('%H:%M')[I_SELL_CANDLE]
    sell_y = trade_window['c'][I_SELL_CANDLE]
    up_sell_candle = trade_window['c'][I_SELL_CANDLE] - trade_window['o'][I_SELL_CANDLE] > 0
    if up_sell_candle:
        dir_sell = -1
    else:
        dir_sell = 1

    fig.add_annotation(
        x=sell_x, y=sell_y,
        ax=0,
        ay=dir_sell * 70,        # long tail downwards
        text="SELL",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor="red",
        font=dict(size=14, color="red")
    )


    fig.update_layout(
        title=dict(
            text=f"{ticker}",
            x=0.5,           # centers the title
            xanchor='center', # ensures centering
            font=dict(size=20) # increase title font size
        ),
        yaxis_title=dict(
            text="Price",
            font=dict(size=16)  # increase y-axis label font size
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0.5,
            dtick=2
        ),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig)

else:
    st.write("No trades to show")
