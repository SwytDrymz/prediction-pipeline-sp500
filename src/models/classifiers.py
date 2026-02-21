import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.models.base import BaseModel


class DecisionTreeClassModel(BaseModel):
    def __init__(
        self, max_depth: int = 5, criterion: str = "gini", threshold: float = 0.005
    ):
        features = ["volume", "log_return", "rsi_14", "atr_14"]
        params = {"max_depth": max_depth, "criterion": criterion}
        super().__init__(
            "DecisionTreeClassifier",
            "classification",
            features,
            params,
            classification_threshold=threshold,
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        lags = 10
        df_features = self.create_lags(df, lags=lags)

        # Use the threshold from self
        df_features["target"] = (
            df_features["log_return"].shift(-1) > self.classification_threshold
        ).astype(int)

        train_df = df_features.dropna(subset=["target"])
        if train_df.empty:
            raise ValueError(
                "Not enough data to train the model after creating lags and target."
            )

        feature_cols = [c for c in df_features.columns if "_lag" in c]
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].to_numpy()

        last_row = df_features.iloc[[-1]]
        X_next = last_row[feature_cols].values

        clf = DecisionTreeClassifier(
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            random_state=42,
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)

        return {
            "prediction": pred,
            "probability": prob,
        }
