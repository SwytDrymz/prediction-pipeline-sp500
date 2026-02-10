import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

from models.base_model import BaseModel

# TODO OPTUNA - bayes optimization for hyperparameters tuning
class DecisionTreeClassModel(BaseModel):
    def __init__(self, max_depth: int = 5, criterion: str = "gini"):
        features = ["volume", "log_return", "sma_20", "sma_5", "atr_14", "rsi_14", "sma_ratio_5_20"]
        params = {"max_depth": max_depth, "criterion": criterion}
        super().__init__("DecisionTreeClassifier", "classification", features, params)

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        lags = 10

        df_features = self.create_lags(df, lags=lags)

        df_features["target"] = (df_features["log_return"].shift(-1) > 0.005).astype(int)

        train_df = df_features.dropna(subset=["target"])

        feature_cols = [c for c in df_features.columns if "_lag" in c]
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
                                                                                                                              
        last_row = df_features.iloc[[-1]]
        X_next = last_row[feature_cols].values

        clf = DecisionTreeClassifier(
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            random_state=42
        )
        clf.fit(X_train, y_train)                                                                                                         
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)

        return {
            "prediction": pred,
            "probability": prob,
        }
