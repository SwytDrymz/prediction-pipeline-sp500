import os
import pandas as pd
import numpy as np
from models.base_model import BaseModel
from typing import Optional

class TradingPipeline:
    def __init__(self, data_dir: str, save_dir: str, eval_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.eval_dir = eval_dir or save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def run(self, model: BaseModel):
        print(f"Starting pipeline with model: {model.name}")
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        eval_results = []

        for file_name in files:
            ticker = file_name.replace(".csv", "")
            
            eval_res = self.evaluate_yesterday(ticker, model)
            if eval_res:
                eval_results.append(eval_res)

            print(f"Processing {ticker} for new prediction...")
            df = pd.read_csv(os.path.join(self.data_dir, file_name))
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            
            if df.empty:
                continue

            pred = model.train_predict_next(df)

            pred_path = os.path.join(self.save_dir, f"{model.model_type}_preds.csv")
            if os.path.exists(pred_path):
                df_preds = pd.read_csv(pred_path, index_col=0)
            else:
                df_preds = pd.DataFrame()

            for key, value in pred.items():
                column_name = f"{ticker}_{model.name}_{key}"
                df_preds.loc[df.index[-1], column_name] = value
            
            df_preds.to_csv(pred_path)

        if eval_results:
            self.save_evaluations(eval_results, model.model_type)

    def save_evaluations(self, results: list, model_type: str):
        eval_file = os.path.join(self.eval_dir, f"{model_type}metrics.csv")
        new_eval_df = pd.DataFrame(results)

        if os.path.exists(eval_file):
            old_eval_df = pd.read_csv(eval_file)
            combined_df = pd.concat([old_eval_df, new_eval_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["ticker", "model", "date"], keep="last", inplace=True)
            combined_df.to_csv(eval_file, index=False)
        else:
            new_eval_df.to_csv(eval_file, index=False)
        
        print(f"Saved {len(results)} evaluation records to {eval_file}")

    def evaluate_yesterday(self, ticker: str, model: BaseModel) -> Optional[dict]:
        pred_path = os.path.join(self.save_dir, f"{model.model_type}_preds.csv")
        data_path = os.path.join(self.data_dir, f"{ticker}.csv")

        if not os.path.exists(pred_path) or not os.path.exists(data_path):
            print(f"Data for {ticker} or predictions not found.")
            return None
        
        df_preds = pd.read_csv(pred_path, index_col=0)
        df_actual = pd.read_csv(data_path)
        
        if "date" in df_actual.columns:
            df_actual.set_index("date", inplace=True)
            
        if len(df_actual) < 2:
            print(f"Not enough data to evaluate {ticker}.")
            return None
            
        today_date = df_actual.index[-1]
        yesterday_date = df_actual.index[-2]
        
        pred_col = f"{ticker}_{model.name}_prediction"
        
        if pred_col not in df_preds.columns or yesterday_date not in df_preds.index:
            print(f"Prediction for {ticker} on {yesterday_date} not found.")
            return None
            
        prediction = df_preds.loc[yesterday_date, pred_col]
        actual_log_return = df_actual.loc[today_date, "log_return"]
        
        result = {
            "ticker": ticker,
            "model": model.name,
            "date": today_date,
            "predicted": prediction,
            "actual_return": actual_log_return
        }
        
        if model.model_type == "classification":
            actual_class = 1 if actual_log_return > 0.005 else 0
            result["actual"] = actual_class
            result["correct"] = int(prediction == actual_class)
        else:
            result["actual"] = actual_log_return
            result["error"] = prediction - actual_log_return
            
        return result