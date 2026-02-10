import requests

import pandas as pd
import numpy as np
from io import StringIO
from typing import Union
from finfetcher import DataFetcher
import pandas_ta as ta

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text

    tables = pd.read_html(StringIO(html))

    sp500_table: Union[pd.DataFrame, None] = None
    for table in tables:
        if "Symbol" in table.columns:
            sp500_table = table
            break

    if sp500_table is None:
        raise ValueError("Could not find S&P 500 table on Wikipedia")

    tickers = sp500_table["Symbol"].tolist()
    return tickers


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 200:
        return df
    df.columns = [c.lower() for c in df.columns]
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["roc_10"] = ta.roc(df["close"], length=10)

    macd = ta.macd(df["close"], fast = 12, slow = 26, signal = 9)
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length = 14)
    bb = ta.bbands(df["close"], length=20, lower_std=2, upper_std=2)
    if bb is not None:
        df["bb_percent"] = (df["close"] - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])

    ema_200 = ta.ema(df["close"], length=200)
    df["dist_ema_200"] = (df["close"] - ema_200) / ema_200

    df["mfi_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length = 14)
    
    return df.dropna()


def update_all_data(tickers: list, data_dir: str):
    for ticker in tickers:
        fetcher = DataFetcher(ticker)
        df = fetcher.get_data(period="max", interval="1d")
        df = add_features(df)

        df.to_csv(f"{data_dir}/{ticker}.csv")