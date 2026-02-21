import requests
import pandas as pd
import numpy as np
from io import StringIO
from typing import Union, Optional
from finfetcher import DataFetcher
import pandas_ta as ta
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers).text
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        logger.error(f"Failed to fetch tickers from Wikipedia: {e}")
        raise

    sp500_table: Union[pd.DataFrame, None] = None
    for table in tables:
        if "Symbol" in table.columns:
            sp500_table = table
            break

    if sp500_table is None:
        raise ValueError("Could not find S&P 500 table on Wikipedia")

    tickers = [
        symbol.strip().replace(".", "-") for symbol in sp500_table["Symbol"].tolist()
    ]
    logger.info(f"Successfully retrieved {len(tickers)} tickers from Wikipedia.")
    return tickers

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 200:
        return df
    df.columns = [c.lower() for c in df.columns]
    
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]    # type: ignore[assignment]
    low: pd.Series = df["low"]      # type: ignore[assignment]
    volume: pd.Series = df["volume"] # type: ignore[assignment]

    df["log_return"] = np.log(close / close.shift(1))
    df["rsi_14"] = ta.rsi(close, length=14)
    df["roc_10"] = ta.roc(close, length=10)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    df["atr_14"] = ta.atr(high, low, close, length=14)
    
    bb = ta.bbands(close, length=20, lower_std=2, upper_std=2)
    if bb is not None:
        l_col = [c for c in bb.columns if c.startswith("BBL")][0]
        u_col = [c for c in bb.columns if c.startswith("BBU")][0]
        df["bb_percent"] = (close - bb[l_col]) / (bb[u_col] - bb[l_col])

    ema_200 = ta.ema(close, length=200)
    if ema_200 is not None:
        df["dist_ema_200"] = (close - ema_200) / ema_200

    df["mfi_14"] = ta.mfi(high, low, close, volume, length=14)

    return df.dropna()

def fetch_ticker_data(ticker: str, period: str = "4y") -> Optional[pd.DataFrame]:
    try:
        fetcher = DataFetcher(ticker)
        df = fetcher.get_data(period=period, interval="1d")
        if df is None or df.empty:
            logger.warning(f"No data returned for ticker: {ticker}")
            return None
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        df = df.rename(columns=column_mapping)
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None
