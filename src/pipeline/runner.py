import pandas as pd
from src.models.base import BaseModel
from src.pipeline.database import DatabaseService
from typing import Optional, List
from src.utils.logging_config import setup_logger
from pandas.tseries.offsets import BDay

logger = setup_logger(__name__)


class TradingPipeline:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def process_ticker(self, ticker: str, df: pd.DataFrame, models: List[BaseModel]):
        if df is None or df.empty or len(df) < 50:
            logger.warning(f"[{ticker}] Received empty DataFrame. Skipping.")
            return

        if "log_return" not in df.columns:
            logger.error(f"[{ticker}] Missing 'log_return' in data. Skipping.")
            return

        # 1. Update Market Data
        df_work = df.copy()
        if "date" in df_work.columns:
            df_work.set_index("date", inplace=True)

        df_work.index = pd.to_datetime(df_work.index)

        latest_db_date = self.db_service.get_latest_date(ticker)
        if latest_db_date:
            new_market_data = df_work[df_work.index > pd.to_datetime(latest_db_date)]
        else:
            new_market_data = df_work

        if not new_market_data.empty:
            self.db_service.save_market_data(new_market_data, ticker)

        # 2. Run Models
        for model in models:
            try:
                today_date_str = df_work.index[-1].strftime("%Y-%m-%d")
                eval_res = self.evaluate_prediction(
                    ticker, df_work, model, today_date_str
                )
                if eval_res:
                    self.db_service.save_evaluation(eval_res, model.model_type)

                # NEW PREDICTION:
                logger.info(f"[{ticker}] Predicting with {model.name}...")
                pred_output = model.train_predict_next(df_work)

                pred_date = df_work.index[-1]
                target_date = (pred_date + BDay(1)).strftime("%Y-%m-%d")
                pred_date_str = pred_date.strftime("%Y-%m-%d")

                pred_record = {
                    "ticker": ticker,
                    "model": model.name,
                    "prediction_date": pred_date_str,
                    "target_date": target_date,
                }

                if model.model_type == "classification":
                    pred_record["predicted_class"] = int(pred_output["prediction"])
                    pred_record["probability"] = float(pred_output["probability"])
                else:
                    pred_record["predicted_return"] = float(pred_output["prediction"])

                self.db_service.save_prediction(pred_record, model.model_type)

            except Exception as e:
                logger.error(f"Error processing {ticker} with {model.name}: {e}")

    def evaluate_prediction(
        self,
        ticker: str,
        df_actual: pd.DataFrame,
        model: BaseModel,
        target_date_str: str,
    ) -> Optional[dict]:
        """
        Looks for a prediction targeting 'target_date_str' and compares it with actual data.
        """
        prev_pred = self.db_service.get_prediction_for_evaluation(
            ticker, model.name, target_date_str, model.model_type
        )
        if not prev_pred:
            return None

        prediction_id = prev_pred["id"]
        predicted_val = prev_pred["predicted_value"]

        target_ts = pd.to_datetime(target_date_str)
        if target_ts not in df_actual.index:
            return None

        actual_log_return = float(df_actual.loc[target_ts, "log_return"])  # type: ignore

        res = {
            "prediction_id": int(prediction_id),
            "ticker": ticker,
            "model": model.name,
            "evaluation_date": target_date_str,
            "actual_return": actual_log_return,
        }

        if model.model_type == "classification":
            actual_class = (
                1 if actual_log_return > model.classification_threshold else 0
            )  # type: ignore
            res.update(
                {
                    "predicted_class": int(predicted_val),
                    "actual_class": int(actual_class),
                    "correct": bool(int(predicted_val) == actual_class),
                }
            )
        else:
            error = predicted_val - actual_log_return
            res.update(
                {
                    "predicted_return": float(predicted_val),
                    "error": float(error),
                    "abs_error": float(abs(error)),
                    "squared_error": float(error**2),
                }
            )

        return res
