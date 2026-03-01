import pandas as pd
from src.models.base import BaseModel, prepare_features
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class LinearRegressionModel(BaseModel):
    def __init__(self, features: list | None = None):
        features_to_init = features if features is not None else []

        super().__init__(
            name="LinearRegression",
            model_type="regression",
            features=features_to_init,
            params={},
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        if not self.features:
            exclude = {"open", "high", "low", "close", "target"}
            self.features = [c for c in df.columns if c.lower() not in exclude]

        X_train, y_train, X_next = prepare_features(df, self.features, 0)

        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_train, y_train)
        pred_next = model.predict(X_next)[0]

        return {
            "prediction": pred_next,
        }
