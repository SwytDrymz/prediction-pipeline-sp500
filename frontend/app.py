import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.pipeline.database import DatabaseService
from src.pipeline.collector import add_features
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def market_data_plot(df, ticker):
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.5, 0.1, 0.2, 0.2],
        subplot_titles=(f"{ticker} Historical Price", "Volume", "RSI (14)", "MACD")
    )

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='Volume',
        marker_color='rgba(100, 150, 255, 0.5)'
    ), row=2, col=1)

    if 'rsi_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['rsi_14'], name='RSI',
            line=dict(color='#FFA500', width=1.5)
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1)

    if 'macd' in df.columns and 'macd_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], name='MACD', line=dict(color='#00E676')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal', line=dict(color='#FF5252')), row=4, col=1)
        
        if 'macd_hist' in df.columns:
            fig.add_trace(go.Bar(x=df['date'], y=df['macd_hist'], name='Histogram', marker_color='gray'), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    return fig
st.set_page_config(page_title="Market Analytics", layout="wide")
st.title("📈 Financial Market Insights")

db = DatabaseService()
available_tickers = db.fetch_available_tickers() 

selected_ticker = st.selectbox("Select Asset Ticker:", available_tickers)

if selected_ticker:
    with st.spinner(f"Loading data for {selected_ticker}..."):
        df_plot = db.fetch_market_data(selected_ticker)
        df_plot = add_features(df_plot)
        
        if not df_plot.empty:
            fig = market_data_plot(df_plot, selected_ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No historical data found for {selected_ticker} in the database.")