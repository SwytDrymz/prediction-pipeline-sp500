import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.models.base import BaseModel, create_lags
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)



class ClassificationModel(BaseModel):
    def __init__(
        self,
        clf_class,
        features: list,
        classification_threshold: float = 0.005,
        **clf_params,
    ):
        super().__init__(
            name=clf_class.__name__,
            model_type="classification",
            features=features,
            params=clf_params,
            classification_threshold=classification_threshold,
        )
        self.clf_class = clf_class
    
    def get_clf(self, y_train=None):
        params = self.params.copy()
        if self.clf_class.__name__ in ["RandomForestClassifier", "DecisionTreeClassifier"]:
            params["class_weight"] = "balanced"
        if self.clf_class.__name__ == "SVC":
            params["probability"] = True
            params["class_weight"] = "balanced"

        if self.clf_class.__name__ == "XGBClassifier" and y_train is not None:
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params["scale_pos_weight"] = pos_weight
        
        clf = self.clf_class(**params)
        
        if self.clf_class.__name__ in ["SVC", "LogisticRegression"]:
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return clf
        
    def _prepare_features(self, df: pd.DataFrame):
            df_work = df.copy()
            lags = 3
            df_features = create_lags(self.features, df_work, lags=lags)
            
            feature_cols_lag = [c for c in df_features.columns if "_lag" in c]
            feature_cols_current = [
                c for c in df_features.columns 
                if c in self.features and c not in ["log_return", "target"]
            ]
            feature_cols = feature_cols_current + feature_cols_lag
            return df_features, feature_cols

    def train_predict_next(self, df: pd.DataFrame) -> dict:
        df_prep, feature_cols = self._prepare_features(df)
        
        df_prep["target"] = (
            df_prep["log_return"].shift(-1) > self.classification_threshold
        ).astype(int)

        train_df = df_prep.iloc[:-1].dropna(subset=["target"] + feature_cols)

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        
        X_next = df_prep[feature_cols].iloc[[-1]].values

        
        clf = self.get_clf(y_train)
        clf.fit(X_train, y_train)
        
        prob = clf.predict_proba(X_next)[0][1]
        return {
            "prediction": int(prob > 0.5),
            "probability": prob,
        }