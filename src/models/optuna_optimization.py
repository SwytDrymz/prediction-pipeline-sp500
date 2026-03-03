import os

import numpy as np
import pandas as pd
import optuna
from dotenv import load_dotenv
from typing import Callable
from sklearn.metrics import f1_score
from functools import partial
from joblib import Parallel, delayed

from src.pipeline.collector import get_sp500_tickers, add_features
from src.pipeline.database import DatabaseService
from src.models.classifiers import ClassificationModel
from src.models.base import create_lags

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

load_dotenv()

db_url = os.getenv("OPTUNA_DB")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

features = ["close", "rsi_14", "roc_10", "volume", "macd_hist", "bb_percent", "dist_ema_200", "volume_rolling_mean_20", "atr_14", "mfi_14"]

THRESHOLD = 0.005

def DecisionTreeClassModel(**params):
    return ClassificationModel(DecisionTreeClassifier, features, THRESHOLD, **params)

def RandomForestClassModel(**params):
    return ClassificationModel(RandomForestClassifier, features, THRESHOLD, **params)

def XGBoostClassModel(**params):
    return ClassificationModel(XGBClassifier, features, THRESHOLD, **params)

def SupportVectorClassModel(**params):
    return ClassificationModel(SVC, features, THRESHOLD, **params)

MODELS = {
    "DecisionTreeClassModel": DecisionTreeClassModel,
    "RandomForestClassModel": RandomForestClassModel,
    "XGBoostClassModel": XGBoostClassModel,
    "SupportVectorClassModel": SupportVectorClassModel,
}



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
    model = model_factory()

    df_prep, feature_cols = model._prepare_features(df)

    
    df_prep["target"] = (df_prep["log_return"].shift(-1) > threshold).astype(int)
    df_prep = df_prep.dropna(subset=["target"] + feature_cols)

    n = len(df_prep)
    test_size = (n - train_window) // n_splits

    if test_size < 20:
        return -999.0

    scores = []

    for i in range(n_splits):
        train_end = train_window + i * test_size
        test_end = min(train_end + test_size, n)

        train_slice = df_prep.iloc[:train_end]
        test_slice = df_prep.iloc[train_end:test_end]

        if len(test_slice) == 0:
            break


        X_train = train_slice[feature_cols].values
        y_train = train_slice["target"].values
        
        X_test = test_slice[feature_cols].values
        y_test = test_slice["target"].values
        clf = model.get_clf(y_train)
        
        try:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            score = f1_score(y_test, preds, zero_division=0)
            scores.append(score)
        except Exception as e:
            logger.error(f"Training failed on split {i}: {e}")
            return -999.0

    return float(np.mean(scores)) if scores else -999.0

def objective(
    trial: optuna.Trial,
    *,
    model_name: str,
    df: pd.DataFrame,
    model_factory: Callable,
    threshold: float,
) -> float:
    params = get_search_space(trial, model_name)
    factory = partial(model_factory, **params)
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
    return df.dropna()



def optimize_ticker(ticker, model_name, factory_fn):
    df = load_data(ticker)
    study = optuna.create_study(
        direction="maximize",
        storage=db_url,
        study_name=f"v2_{ticker}_{model_name}",
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, model_name=model_name, df=df,
                model_factory=factory_fn, threshold=THRESHOLD),
        n_trials=50,
        n_jobs=1,
    )

def main():
    tickers = get_sp500_tickers()
    for model_name, factory_fn in MODELS.items():
        Parallel(n_jobs=1)(
            delayed(optimize_ticker)(ticker, model_name, factory_fn)
            for ticker in tickers
        )

if __name__ == "__main__":
    main()