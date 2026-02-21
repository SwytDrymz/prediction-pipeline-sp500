import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from src.utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)


def initialize_database():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not found in environment.")
        return

    engine = create_engine(db_url)

    tables = [
        # Market Data
        """
        CREATE TABLE IF NOT EXISTS market_data (
            date DATE NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (date, ticker)
        );  
        """,
        # Predictions Classification
        """
        CREATE TABLE IF NOT EXISTS predictions_classification (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            model VARCHAR(50) NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            predicted_class INT NOT NULL,
            probability FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, model, prediction_date)
        );
        """,
        # Predictions Regression
        """
        CREATE TABLE IF NOT EXISTS predictions_regression (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            model VARCHAR(50) NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            predicted_return FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, model, prediction_date)
        );
        """,
        # Evaluations Classification
        """
        CREATE TABLE IF NOT EXISTS evaluations_classification (
            id SERIAL PRIMARY KEY,
            prediction_id INT NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            model VARCHAR(50) NOT NULL,
            evaluation_date DATE NOT NULL,
            predicted_class INT NOT NULL,
            actual_class INT NOT NULL,
            correct BOOLEAN NOT NULL,
            actual_return FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (prediction_id) REFERENCES predictions_classification(id) ON DELETE CASCADE,
            UNIQUE(prediction_id)
        );
        """,
        # Evaluations Regression
        """
        CREATE TABLE IF NOT EXISTS evaluations_regression (
            id SERIAL PRIMARY KEY,
            prediction_id INT NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            model VARCHAR(50) NOT NULL,
            evaluation_date DATE NOT NULL,
            predicted_return FLOAT NOT NULL,
            actual_return FLOAT NOT NULL,
            error FLOAT NOT NULL,
            abs_error FLOAT NOT NULL,
            squared_error FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (prediction_id) REFERENCES predictions_regression(id) ON DELETE CASCADE,
            UNIQUE(prediction_id)
        );
        """,
    ]

    indices = [
        "CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date);",
        "CREATE INDEX IF NOT EXISTS idx_pred_class_ticker_model ON predictions_classification(ticker, model, prediction_date);",
        "CREATE INDEX IF NOT EXISTS idx_pred_reg_ticker_model ON predictions_regression(ticker, model, prediction_date);",
        "CREATE INDEX IF NOT EXISTS idx_eval_class_ticker_model ON evaluations_classification(ticker, model, evaluation_date);",
        "CREATE INDEX IF NOT EXISTS idx_eval_reg_ticker_model ON evaluations_regression(ticker, model, evaluation_date);",
        # New indexes for target_date performance
        "CREATE INDEX IF NOT EXISTS idx_pred_class_target_date ON predictions_classification(ticker, model, target_date);",
        "CREATE INDEX IF NOT EXISTS idx_pred_reg_target_date ON predictions_regression(ticker, model, target_date);",
    ]

    with engine.connect() as conn:
        for ddl in tables:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception as e:
                logger.error(f"Error creating table: {e}")

        for idx_ddl in indices:
            try:
                conn.execute(text(idx_ddl))
                conn.commit()
            except Exception as e:
                logger.error(f"Error creating index: {e}")

    logger.info("Database schema initialized.")


if __name__ == "__main__":
    initialize_database()
