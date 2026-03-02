import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.models.base import BaseModel, prepare_features, create_lags
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class DecisionTreeClassModel(BaseModel):
    def __init__(
        self,
        features: list | None = None,
        max_depth: int = 5,
        criterion: str = "gini",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        threshold: float = 0.005,
    ):
        features_to_init = features if features is not None else []

        params = {
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
        }

        super().__init__(
            name="DecisionTreeClassifier",
            model_type="classification",
            features=features_to_init,
            params=params,
            classification_threshold=threshold,
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        if not self.features:
            exclude = {"open", "high", "low", "close", "target"}
            self.features = [c for c in df.columns if c.lower() not in exclude]

        X_train, y_train, X_next = prepare_features(
            df, self.features, self.classification_threshold
        )

        clf = DecisionTreeClassifier(
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            class_weight="balanced",
            min_samples_leaf=self.params["min_samples_leaf"],
            min_samples_split=self.params["min_samples_split"],
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)

        return {
            "prediction": pred,
            "probability": prob,
        }
    
    def fit_predict_batch(self, train_df, test_df):
        feature_cols = [c for c in train_df.columns if c != "log_return"]

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        
        X_test = test_df[feature_cols].values
        clf = DecisionTreeClassifier(
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            class_weight="balanced",
            min_samples_leaf=self.params["min_samples_leaf"],
            min_samples_split=self.params["min_samples_split"],
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)


class RandomForestClassModel(BaseModel):
    def __init__(
        self,
        features: list | None = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        n_estimators: int = 200,
        max_depth: int = 5,
        criterion: str = "gini",
        threshold: float = 0.005,
    ):
        features_to_init = features if features is not None else []

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
        }

        super().__init__(
            name="RandomForestClassifier",
            model_type="classification",
            features=features_to_init,
            params=params,
            classification_threshold=threshold,
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        if not self.features:
            exclude = {"open", "high", "low", "close", "target"}
            self.features = [c for c in df.columns if c.lower() not in exclude]

        X_train, y_train, X_next = prepare_features(
            df, self.features, self.classification_threshold
        )

        clf = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            criterion=self.params["criterion"],
            class_weight="balanced",
            min_samples_leaf=self.params["min_samples_leaf"],
            min_samples_split=self.params["min_samples_split"],
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)
        return {
            "prediction": pred,
            "probability": prob,
        }
    def fit_predict_batch(self, train_df, test_df):
        feature_cols = [c for c in train_df.columns if c != "log_return"]

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        
        X_test = test_df[feature_cols].values
        clf = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            criterion=self.params["criterion"],
            class_weight="balanced",
            min_samples_leaf=self.params["min_samples_leaf"],
            min_samples_split=self.params["min_samples_split"],
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)


class XGBoostClassModel(BaseModel):
    def __init__(
        self,
        features: list | None = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        threshold: float = 0.005,
    ):
        features_to_init = features if features is not None else []

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

        super().__init__(
            name="XGBoostClassifier",
            model_type="classification",
            features=features_to_init,
            params=params,
            classification_threshold=threshold,
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        if not self.features:
            exclude = {"open", "high", "low", "close", "target"}
            self.features = [c for c in df.columns if c.lower() not in exclude]

        X_train, y_train, X_next = prepare_features(
            df, self.features, self.classification_threshold
        )

        clf = XGBClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)
        return {
            "prediction": pred,
            "probability": prob,
        }
    def fit_predict_batch(self, train_df, test_df):
        feature_cols = [c for c in train_df.columns if c != "log_return"]
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        
        X_test = test_df[feature_cols].values
        clf = XGBClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            tree_method="hist",
            device="cuda",
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)


class SupportVectorClassModel(BaseModel):
    def __init__(
        self,
        features: list | None = None,
        gamma: str = "scale",
        kernel: str = "rbf",
        C: float = 1.0,
        threshold: float = 0.005,
    ):
        features_to_init = features if features is not None else []

        params = {"kernel": kernel, "C": C, "gamma": gamma}

        super().__init__(
            name="SupportVectorClassifier",
            model_type="classification",
            features=features_to_init,
            params=params,
            classification_threshold=threshold,
        )

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        if not self.features:
            exclude = {"open", "high", "low", "close", "target"}
            self.features = [c for c in df.columns if c.lower() not in exclude]

        X_train, y_train, X_next = prepare_features(
            df, self.features, self.classification_threshold
        )

        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel=self.params["kernel"],
                        C=self.params["C"],
                        gamma=self.params["gamma"],
                        probability=True,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_next)[0][1]
        pred = int(prob > 0.5)
        return {
            "prediction": pred,
            "probability": prob,
        }
    def fit_predict_batch(self, train_df, test_df):
        feature_cols = [c for c in train_df.columns if c != "log_return"]
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        
        X_test = test_df[feature_cols].values
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel=self.params["kernel"],
                        C=self.params["C"],
                        gamma=self.params["gamma"],
                        probability=True,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)