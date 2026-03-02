import os

import numpy as np
import pandas as pd
import optuna
from dotenv import load_dotenv
from typing import Callable
from sklearn.metrics import f1_score

from src.pipeline.collector import get_sp500_tickers, add_features
from src.pipeline.database import DatabaseService
from src.models.classifiers import (
    DecisionTreeClassModel,
    RandomForestClassModel,
    XGBoostClassModel,
    SupportVectorClassModel,
)
from src.models.base import create_lags

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

load_dotenv()

db_url = os.getenv("OPTUNA_DB")


MODELS = {
    "DecisionTreeClassModel": DecisionTreeClassModel,
    "RandomForestClassModel": RandomForestClassModel,
    "XGBoostClassModel": XGBoostClassModel,
    "SupportVectorClassModel": SupportVectorClassModel,
}
THRESHOLD = 0.005


def get_search_space(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "DecisionTreeClassModel":
        return {
            "max_depth": trial.suggest_int("dt_max_depth", 3, 10),
            "criterion": trial.suggest_categorical("dt_criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_int("dt_min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("dt_min_samples_split", 2, 10),
        }
    elif model_name == "RandomForestClassModel":
        return {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 10),
            "criterion": trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        }
    elif model_name == "XGBoostClassModel":
        return {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
        }
    elif model_name == "SupportVectorClassModel":
        return {
            "kernel": trial.suggest_categorical("svc_kernel", ["linear", "rbf"]),
            "C": trial.suggest_float("svc_C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical("svc_gamma", ["scale", "auto"]),
        }
    return {}

def walk_forward_score(
    df: pd.DataFrame,
    model_factory: Callable,
    threshold: float = 0.005,
    train_window: int = 252,
    n_splits: int = 5,
) -> float:
    n = len(df)
    test_size = (n - train_window) // n_splits
    
    if test_size < 20:
        return -999.0

    scores = []

    for i in range(n_splits):
        train_start = i * test_size
        train_end = train_start + train_window
        test_end = min(train_end + test_size, n)

        train_slice = df.iloc[train_start:train_end].copy()
        test_slice = df.iloc[train_end:test_end].copy()

        if len(test_slice) == 0:
            break

        try:
            model = model_factory()
            model.classification_threshold = threshold
            
            y_pred = model.fit_predict_batch(train_slice, test_slice)
            
            y_true = (test_slice["log_return"] > threshold).astype(int).values
            
            if len(y_true) > 0:
                score = f1_score(y_true, y_pred, zero_division=0) # type: ignore
                scores.append(score)
                
        except Exception as e:
            print(f"Error v foldu {i}: {e}")
            continue

    return float(np.mean(scores)) if scores else -999.0

def objective(
    trial: optuna.Trial,
    model_name: str,
    df: pd.DataFrame,
    model_factory: Callable,
    threshold: float,
) -> float:
    paraams = get_search_space(trial, model_name)
    factory = lambda: model_factory(**paraams)
    return walk_forward_score(df, factory, threshold)


def load_data(ticker: str) -> pd.DataFrame:
    db = DatabaseService()
    df = db.fetch_market_data(ticker)
    df_work = df.copy()
    if "date" in df_work.columns:
        df_work.set_index("date", inplace=True)
    if "ticker" in df_work.columns:
        df_work.drop(columns=["ticker"], inplace=True)

    df_work.index = pd.to_datetime(df_work.index)
    df = add_features(df_work)
    df = create_lags(list(df.columns), df, lags=3)

    df["target"] = (df["log_return"].shift(-1) > THRESHOLD).astype(int)


    df = df.iloc[:-1600]
    return df.dropna()


def main():
    tickers = get_sp500_tickers()

    for model_name, factory_fn in MODELS.items():
        for ticker in tickers:
            df = load_data(ticker)
            study = optuna.create_study(
                direction="maximize",
                storage=db_url,
                study_name=f"v2_{ticker}_{model_name}",
                load_if_exists=True,
            )
            study.optimize(
                lambda trial, m=model_name,f=factory_fn: objective(
                    trial, m, df, f, THRESHOLD
                ),
                n_trials=50,
                n_jobs=-1
            )
if __name__ == "__main__":
    main()