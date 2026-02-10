from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional


class BaseModel(ABC):
    def __init__(
        self, name: str, model_type: str, features: list, params: Optional[dict] = None
    ):
        """
        name: model name
        model_type: 'classification' or 'regression'
        features: list of feature column names to be used
        params: dictionary of model parameters
        """
        self.name = name
        self.model_type = model_type
        self.features = features
        self.params = params if params else {}

        print(f"Initialized model: {self.name} of type {self.model_type}")

    def create_lags(self, df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        """Common method to create lagged features."""

        df_lags = df.copy()

        for lag in range(1, lags + 1):
            for col in self.features:
                if col in df_lags.columns:
                    df_lags[f"{col}_lag{lag}"] = df_lags[col].shift(lag)
        df_lags.dropna(inplace=True)
        return df_lags

    @abstractmethod
    def train_predict_next(self, df: pd.DataFrame) -> dict:
        """
        trains model on history and predicts next day return or class.

        Args:
            df: DataFrame with data up to and including today.
                                                                                                                    █
        Returns:
            dict: {
                "prediction": int/float,
                "probability": float (volitelné),
            }
        """
        pass