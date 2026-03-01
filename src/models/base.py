from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_lags(features: list, df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    df_lags = df.copy()
    for lag in range(1, lags + 1):
        for col in features:
            if col in df_lags.columns:
                df_lags[f"{col}_lag{lag}"] = df_lags[col].shift(lag)
    df_lags.dropna(inplace=True)
    return df_lags


def prepare_features(
    df: pd.DataFrame, features: list, class_threshold: float | None
) -> tuple:
    df = df.copy()
    lags = 3
    df_features = create_lags(features, df, lags=lags)

    if class_threshold is not None:
        df_features["target"] = (
            df_features["log_return"].shift(-1) > class_threshold
        ).astype(int)
    else:
        df_features["target"] = df_features["log_return"].shift(-1)

    last_row = df_features.iloc[[-1]]
    train_df = df_features.iloc[:-1].copy()
    feature_cols_lag = [c for c in df_features.columns if "_lag" in c]
    feature_cols_current = [
        c for c in df_features.columns if c in features and c != "log_return"
    ]

    feature_cols = feature_cols_current + feature_cols_lag

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].to_numpy()
    X_next = last_row[feature_cols].values
    return X_train, y_train, X_next


class BaseModel(ABC):
    def __init__(
        self,
        name: str,
        model_type: str,
        features: list,
        params: Optional[dict] = None,
        classification_threshold: float = 0.005,
    ):
        self.name = name
        self.model_type = model_type
        self.features = features
        self.params = params if params else {}
        self.classification_threshold = classification_threshold

        logger.info(f"Initialized model: {self.name} of type {self.model_type}")

    @abstractmethod
    def train_predict_next(self, df: pd.DataFrame) -> dict:
        pass
