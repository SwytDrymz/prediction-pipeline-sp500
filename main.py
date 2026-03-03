from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from src.pipeline.collector import get_sp500_tickers, fetch_ticker_data, add_features
from src.pipeline.runner import TradingPipeline
from src.pipeline.database import DatabaseService
from src.models.classifiers import ClassificationModel
from src.utils.logging_config import setup_logger
from src.models.base import BaseModel

import pandas as pd

logger = setup_logger("main")


def main():
    logger.info("Starting pipeline execution.")

    db_service = DatabaseService()
    if db_service.engine is None:
        logger.error("Database connection is required. Check DATABASE_URL.")
        return

    pipeline = TradingPipeline(db_service=db_service)
    features = ["close", "rsi_14", "roc_10", "volume", "macd_hist", "bb_percent", "dist_ema_200", "volume_rolling_mean_20", "atr_14", "mfi_14"]
    models: list[BaseModel] = [
        ClassificationModel(DecisionTreeClassifier, features=features, max_depth=5, criterion="gini"),
        ClassificationModel(RandomForestClassifier, features=features, n_estimators=200, max_depth=5, criterion="gini"),
        ClassificationModel(XGBClassifier, features=features, n_estimators=200, max_depth=5, learning_rate=0.1),
        ClassificationModel(SVC, features=features, kernel="rbf", C=1.0, gamma="scale"),
        #ClassificationModel(LinearRegressionModel, features=["open", "high", "low", "close", "volume"]),
    ]

    logger.info("Fetching S&P 500 tickers...")
    try:
        tickers = get_sp500_tickers()
    except Exception as e:
        logger.critical(f"Critical error fetching tickers: {e}")
        return

    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        new_df = fetch_ticker_data(ticker)
        if new_df is None or new_df.empty:
            logger.warning(f"[{ticker}] No data fetched. Skipping.")
            continue
        db_df = db_service.fetch_market_data(ticker)

        if not db_df.empty:
            db_df = db_df[["date", "open", "high", "low", "close", "volume"]]
            combined_df = (
                pd.concat([db_df, new_df])
                .drop_duplicates(subset="date")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            combined_df = new_df

        combined_df = add_features(combined_df)

        if not combined_df.empty:
            pipeline.process_ticker(ticker, combined_df, models)
        else:
            logger.warning(
                f"[{ticker}] No valid data after feature engineering. Skipping."
            )


if __name__ == "__main__":
    main()
