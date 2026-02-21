import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.dialects.postgresql import insert
from src.utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class DatabaseService:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL not found in environment.")
            self.engine = None
            self.metadata = None
            return

        self.engine = create_engine(db_url)

        self.metadata = MetaData()
        try:
            self.metadata.reflect(bind=self.engine)
            logger.info("Database metadata loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load database metadata: {e}")
            self.metadata = None

    def save_market_data(self, df: pd.DataFrame, ticker: str):
        if self.engine is None or self.metadata is None or df.empty:
            return

        df_to_save = df.reset_index() if "date" not in df.columns else df.copy()
        df_to_save = df_to_save[["date", "open", "high", "low", "close", "volume"]]
        df_to_save["ticker"] = ticker

        records: list[dict] = df_to_save.to_dict(orient="records")  # type: ignore[assignment]

        try:
            with self.engine.begin() as conn:
                table = self.metadata.tables["market_data"]
                stmt = (
                    insert(table)
                    .values(records)
                    .on_conflict_do_nothing(index_elements=["date", "ticker"])
                )
                conn.execute(stmt)
            logger.debug(f"[{ticker}] Saved {len(records)} market data records.")
        except Exception as e:
            logger.error(f"[{ticker}] Failed to save market data: {e}")

    def save_prediction(self, record: dict, model_type: str):
        if self.engine is None or self.metadata is None:
            return

        table_name = f"predictions_{model_type}"

        try:
            with self.engine.begin() as conn:
                table = self.metadata.tables[table_name]
                stmt = (
                    insert(table)
                    .values(record)
                    .on_conflict_do_nothing(
                        index_elements=["ticker", "model", "prediction_date"]
                    )
                )
                conn.execute(stmt)
            logger.debug(f"Saved {model_type} prediction for {record.get('ticker')}.")
        except Exception as e:
            logger.error(f"Failed to save {model_type} prediction: {e}")

    def save_evaluation(self, record: dict, model_type: str):
        if self.engine is None or self.metadata is None:
            return

        table_name = f"evaluations_{model_type}"

        try:
            with self.engine.begin() as conn:
                table = self.metadata.tables[table_name]
                stmt = (
                    insert(table)
                    .values(record)
                    .on_conflict_do_nothing(index_elements=["prediction_id"])
                )
                conn.execute(stmt)
            logger.debug(
                f"Saved {model_type} evaluation for prediction {record.get('prediction_id')}."
            )
        except Exception as e:
            logger.error(f"Failed to save {model_type} evaluation: {e}")

    def get_prediction_for_evaluation(
        self, ticker: str, model: str, target_date: str, model_type: str
    ):
        if self.engine is None:
            return None

        table_name = f"predictions_{model_type}"
        val_col = (
            "predicted_class" if model_type == "classification" else "predicted_return"
        )

        query = text(f"""
            SELECT id, {val_col} as predicted_value 
            FROM {table_name} 
            WHERE ticker = :ticker AND model = :model AND target_date = :target_date
            LIMIT 1
        """)

        try:
            with self.engine.connect() as conn:
                res = pd.read_sql(
                    query,
                    conn,
                    params={
                        "ticker": ticker,
                        "model": model,
                        "target_date": target_date,
                    },
                )
                return res.iloc[0].to_dict() if not res.empty else None
        except Exception as e:
            logger.error(f"Error fetching prediction for evaluation: {e}")
            return None

    def get_latest_date(self, ticker: str):
        if self.engine is None:
            return None

        query = text("SELECT max(date) FROM market_data WHERE ticker = :ticker")

        try:
            with self.engine.connect() as conn:
                return conn.execute(query, {"ticker": ticker}).scalar()
        except Exception as e:
            logger.error(f"Error fetching latest date for {ticker}: {e}")
            return None
