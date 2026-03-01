import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from frontend.scripts.plots import (
    plot_classification_overall_winrate,
    plot_regression_overall_error,
)

import streamlit as st
from src.pipeline.database import DatabaseService
import pandas as pd

db_service = DatabaseService()


@st.cache_data(ttl=3600)
def get_eval_data(model_type: str) -> pd.DataFrame:
    df = db_service.fetch_evaluations_for_type(model_type)
    if not df.empty:
        df["evaluation_date"] = pd.to_datetime(df["evaluation_date"])
        df = df.sort_values("evaluation_date")
    return df


@st.cache_data
def get_tickers():
    return db_service.fetch_available_tickers()


def main():
    st.set_page_config(page_title="Bot Evaluace", layout="wide")

    class_eval_df = get_eval_data("classification")
    reg_eval_df = get_eval_data("regression")

    if class_eval_df.empty:
        st.warning("No classification evaluation data available.")
        return

    st.title("Model Performance Dashboard")
    row1_col1, row1_col2 = st.columns(2)

    # CLASS WR
    with row1_col1:
        st.subheader("Classification Winrate")
        plot_classification_overall_winrate(class_eval_df)

    # REG ERROR
    with row1_col2:
        st.subheader("Regression Error")
        plot_regression_overall_error(reg_eval_df)

    st.divider()
    row2_col1, row2_col2 = st.columns(2)

    # CLASS METRICS
    with row2_col1:
        pass

    # REG METRICS
    with row2_col2:
        pass

    st.divider()

    row_3_col1, row_3_col2 = st.columns(2)

    # BEST/WORST MODELS (CLASS & REG)
    with row_3_col1:
        pass

    # BEST/WORST TICKERS OVERALL
    with row_3_col2:
        pass


if __name__ == "__main__":
    main()
