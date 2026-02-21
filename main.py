from src.pipeline.collector import get_sp500_tickers, fetch_ticker_data, add_features
from src.pipeline.runner import TradingPipeline
from src.pipeline.database import DatabaseService
from src.models.classifiers import DecisionTreeClassModel
from src.utils.logging_config import setup_logger
from src.models.base import BaseModel

logger = setup_logger("main")


def main():
    logger.info("Starting pipeline execution.")

    db_service = DatabaseService()
    if db_service.engine is None:
        logger.error("Database connection is required. Check DATABASE_URL.")
        return

    pipeline = TradingPipeline(db_service=db_service)

    models: list[BaseModel] = [
        DecisionTreeClassModel(max_depth=5, criterion="gini"),
    ]

    logger.info("Fetching S&P 500 tickers...")
    try:
        tickers = get_sp500_tickers()
    except Exception as e:
        logger.critical(f"Critical error fetching tickers: {e}")
        return

    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        df = fetch_ticker_data(ticker)
        if df is not None:
            df = add_features(df)
            pipeline.process_ticker(ticker, df, models)
        else:
            logger.warning(f"No data for {ticker}. Skipping.")


if __name__ == "__main__":
    main()
