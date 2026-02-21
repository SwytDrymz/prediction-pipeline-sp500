from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


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

    def create_lags(self, df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        df_lags = df.copy()
        for lag in range(1, lags + 1):
            for col in self.features:
                if col in df_lags.columns:
                    df_lags[f"{col}_lag{lag}"] = df_lags[col].shift(lag)
        df_lags.dropna(inplace=True)
        return df_lags

    @abstractmethod
    def train_predict_next(self, df: pd.DataFrame) -> dict:
        pass
