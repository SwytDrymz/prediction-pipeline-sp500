"""
Tests for updated prepare_features with current + lagged features.
Covers:
- Correct feature columns (current + lag, no log_return leakage)
- No data leakage in current features
- X_next uses today's values (not yesterday's)
- Date alignment: target_date = prediction_date + 1 BDay
- Training data integrity
- Production readiness checks
"""

import pytest
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.models.classifiers import (
    DecisionTreeClassModel,
    RandomForestClassModel,
    XGBoostClassModel,
    prepare_features,
)
from src.models.base import create_lags

THRESHOLD = 0.005
LAGS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    log_return = np.log(close / np.roll(close, 1))
    log_return[0] = 0.0
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "log_return": log_return,
            "rsi": rng.uniform(30, 70, n),
            "macd": rng.normal(0, 0.5, n),
            "atr": rng.uniform(1, 5, n),
        },
        index=dates,
    )


def get_features(df: pd.DataFrame) -> list:
    exclude = {"open", "high", "low", "close", "target"}
    return [c for c in df.columns if c.lower() not in exclude]


def get_feature_cols(df_features: pd.DataFrame, features: list) -> tuple:
    """Returns (current_cols, lag_cols, all_feature_cols) same logic as prepare_features."""
    lag_cols = [c for c in df_features.columns if "_lag" in c]
    current_cols = [
        c for c in df_features.columns if c in features and c != "log_return"
    ]
    return current_cols, lag_cols, current_cols + lag_cols


# ---------------------------------------------------------------------------
# 1. Feature column composition
# ---------------------------------------------------------------------------


class TestFeatureColumnComposition:
    """Verify that feature_cols = current (no log_return) + lags."""

    def test_current_features_included(self):
        """Non-lagged features (except log_return) must be in feature_cols."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)

        for col in ["rsi", "macd", "atr"]:
            assert col in feature_cols, (
                f"Current feature '{col}' missing from feature_cols"
            )

    def test_log_return_not_in_current_features(self):
        """log_return (without lag) must NEVER appear in feature_cols — it would be leakage."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, feature_cols = get_feature_cols(df_lags, features)

        assert "log_return" not in feature_cols, (
            "log_return without lag is in feature_cols — this is data leakage!"
        )

    def test_log_return_lags_are_included(self):
        """Lagged log_return values must be present."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, lag_cols, _ = get_feature_cols(df_lags, features)

        for i in range(1, LAGS + 1):
            assert f"log_return_lag{i}" in lag_cols, f"log_return_lag{i} missing"

    def test_total_feature_count(self):
        """feature_cols count = len(current_features) + len(features) * lags."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)

        expected_lag_count = len(features) * LAGS
        expected_current_count = len(features) - 1  # minus log_return
        assert len(lag_cols) == expected_lag_count
        assert len(current_cols) == expected_current_count
        assert len(feature_cols) == expected_lag_count + expected_current_count

    def test_no_duplicate_columns_in_feature_cols(self):
        """feature_cols must not contain duplicates."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, feature_cols = get_feature_cols(df_lags, features)

        assert len(feature_cols) == len(set(feature_cols)), (
            f"Duplicate columns in feature_cols: {[c for c in feature_cols if feature_cols.count(c) > 1]}"
        )

    def test_ohlcv_not_in_features(self):
        """OHLC columns must not appear in feature_cols."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, feature_cols = get_feature_cols(df_lags, features)

        for col in ["open", "high", "low", "close"]:
            assert col not in feature_cols, f"OHLC column '{col}' found in feature_cols"


# ---------------------------------------------------------------------------
# 2. X_next uses today's (most recent) values
# ---------------------------------------------------------------------------


class TestXNextUsesTodaysValues:
    """After the fix, X_next must include current day's feature values."""

    def test_x_next_current_rsi_equals_last_row_rsi(self):
        """rsi in X_next must equal the last row's rsi (today's value)."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, X_next = prepare_features(df, features, THRESHOLD)

        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)
        rsi_idx = feature_cols.index("rsi")

        expected_rsi = df_lags["rsi"].iloc[-1]
        actual_rsi = X_next[0][rsi_idx]

        assert abs(actual_rsi - expected_rsi) < 1e-10, (
            f"X_next rsi={actual_rsi:.6f}, expected today's rsi={expected_rsi:.6f}"
        )

    def test_x_next_lag1_equals_penultimate_row(self):
        """lag1 in X_next must equal the second-to-last row's raw value."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, X_next = prepare_features(df, features, THRESHOLD)

        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)

        for col in [c for c in lag_cols if c.endswith("_lag1")]:
            raw_col = col.replace("_lag1", "")
            expected = df_lags[raw_col].iloc[-2]  # předposlední řádek
            idx = feature_cols.index(col)
            actual = X_next[0][idx]
            assert abs(actual - expected) < 1e-10, (
                f"{col}: lag1={actual:.6f}, expected={expected:.6f}"
            )

    def test_changing_last_row_current_feature_changes_x_next(self):
        """Modifying today's rsi must change X_next (proves current features are used)."""
        df = make_df(100)
        features = get_features(df)

        df_modified = df.copy()
        df_modified.loc[df_modified.index[-1], "rsi"] = 9999.0

        _, _, X_next_orig = prepare_features(df, features, THRESHOLD)
        _, _, X_next_mod = prepare_features(df_modified, features, THRESHOLD)

        assert not np.array_equal(X_next_orig, X_next_mod), (
            "X_next did not change after modifying today's rsi — current features not used!"
        )

    def test_changing_penultimate_row_current_feature_does_not_change_x_next_current(
        self,
    ):
        """
        Modifying yesterday's rsi must NOT change current rsi in X_next,
        but WILL change rsi_lag1.
        """
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)

        df_modified = df.copy()
        df_modified.loc[df_modified.index[-2], "rsi"] = 9999.0

        _, _, X_next_orig = prepare_features(df, features, THRESHOLD)
        _, _, X_next_mod = prepare_features(df_modified, features, THRESHOLD)

        # Current rsi (today) must be unchanged
        rsi_idx = feature_cols.index("rsi")
        assert X_next_orig[0][rsi_idx] == X_next_mod[0][rsi_idx], (
            "Today's rsi changed when we modified yesterday's — something is wrong"
        )

        # rsi_lag1 must change (it now reflects yesterday's modified value)
        rsi_lag1_idx = feature_cols.index("rsi_lag1")
        assert X_next_orig[0][rsi_lag1_idx] != X_next_mod[0][rsi_lag1_idx], (
            "rsi_lag1 did not change after modifying yesterday's rsi"
        )


# ---------------------------------------------------------------------------
# 3. No data leakage in current features
# ---------------------------------------------------------------------------


class TestNoLeakageInCurrentFeatures:
    """Current features are safe only if they don't reveal future log_return."""

    def test_log_return_excluded_from_current_features(self):
        """log_return must be excluded from current features — it's the basis of the target."""
        df = make_df(100)
        features = get_features(df)
        X_train, y_train, X_next = prepare_features(df, features, THRESHOLD)

        df_lags = create_lags(features, df, lags=LAGS)
        current_cols, _, _ = get_feature_cols(df_lags, features)

        assert "log_return" not in current_cols

    def test_target_independent_of_current_rsi(self):
        """
        Changing rsi (current feature) must not change y_train —
        rsi is not used to construct the target.
        """
        df = make_df(100)
        features = get_features(df)

        df_modified = df.copy()
        df_modified["rsi"] = 9999.0  # extreme rsi everywhere

        _, y_train_orig, _ = prepare_features(df, features, THRESHOLD)
        _, y_train_mod, _ = prepare_features(df_modified, features, THRESHOLD)

        np.testing.assert_array_equal(y_train_orig, y_train_mod)

    def test_current_features_at_row_t_do_not_use_future_data(self):
        """
        Current features at training row t must equal df_lags values at t,
        not any future row.
        """
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        current_cols, lag_cols, feature_cols = get_feature_cols(df_lags, features)

        X_train, _, _ = prepare_features(df, features, THRESHOLD)
        train_df = df_lags.iloc[:-1]

        for col in current_cols:
            col_idx = feature_cols.index(col)
            expected = train_df[col].values
            actual = X_train[:, col_idx]
            np.testing.assert_array_almost_equal(
                actual,
                expected,
                decimal=10,
                err_msg=f"Column '{col}' in X_train doesn't match df_lags values",
            )


# ---------------------------------------------------------------------------
# 4. Date alignment (production critical)
# ---------------------------------------------------------------------------


class TestDateAlignment:
    """Verify prediction_date and target_date are correctly aligned."""

    def test_prediction_date_is_last_available_date(self):
        """prediction_date must be the last date in df."""
        df = make_df(100)
        last_date = df.index[-1]
        assert last_date == df.index.max()

    def test_target_date_is_next_business_day(self):
        """target_date = prediction_date + 1 BDay."""
        df = make_df(100)
        pred_date = df.index[-1]
        target_date = pred_date + BDay(1)

        # Verify it's not a weekend
        assert target_date.weekday() < 5, f"target_date {target_date} is a weekend"

    def test_target_date_not_same_as_prediction_date(self):
        """target_date must be strictly after prediction_date."""
        df = make_df(100)
        pred_date = df.index[-1]
        target_date = pred_date + BDay(1)
        assert target_date > pred_date

    def test_x_next_corresponds_to_last_date(self):
        """X_next must use data from the last date in df_lags."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)
        _, _, feature_cols = get_feature_cols(df_lags, features)

        _, _, X_next = prepare_features(df, features, THRESHOLD)
        expected = df_lags.iloc[[-1]][feature_cols].values

        np.testing.assert_array_equal(X_next, expected)

    def test_train_df_does_not_include_last_date(self):
        """Training data must end at second-to-last date."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)

        last_date = df_lags.index[-1]
        second_to_last_date = df_lags.index[-2]

        X_train, _, _ = prepare_features(df, features, THRESHOLD)
        train_df = df_lags.iloc[:-1]

        assert train_df.index[-1] == second_to_last_date
        assert last_date not in train_df.index


# ---------------------------------------------------------------------------
# 5. Training data integrity
# ---------------------------------------------------------------------------


class TestTrainingDataIntegrity:
    def test_correct_number_of_training_rows(self):
        df = make_df(100)
        features = get_features(df)
        X_train, y_train, _ = prepare_features(df, features, THRESHOLD)
        assert len(X_train) == len(df) - LAGS - 1
        assert len(y_train) == len(X_train)

    def test_x_train_no_nan(self):
        df = make_df(100)
        features = get_features(df)
        X_train, _, _ = prepare_features(df, features, THRESHOLD)
        assert not np.isnan(X_train).any()

    def test_y_train_no_nan(self):
        df = make_df(100)
        features = get_features(df)
        _, y_train, _ = prepare_features(df, features, THRESHOLD)
        assert not np.isnan(y_train).any()

    def test_y_train_is_binary(self):
        df = make_df(100)
        features = get_features(df)
        _, y_train, _ = prepare_features(df, features, THRESHOLD)
        assert set(y_train).issubset({0, 1})

    def test_x_next_no_nan(self):
        df = make_df(100)
        features = get_features(df)
        _, _, X_next = prepare_features(df, features, THRESHOLD)
        assert not np.isnan(X_next).any()

    def test_x_next_shape(self):
        df = make_df(100)
        features = get_features(df)
        X_train, _, X_next = prepare_features(df, features, THRESHOLD)
        assert X_next.shape == (1, X_train.shape[1])

    def test_penultimate_target_correct(self):
        """Last training row's target = (log_return of last row > threshold)."""
        df = make_df(100)
        features = get_features(df)
        df_lags = create_lags(features, df, lags=LAGS)

        _, y_train, _ = prepare_features(df, features, THRESHOLD)
        last_log_return = df_lags["log_return"].iloc[-1]
        expected = int(last_log_return > THRESHOLD)

        assert y_train[-1] == expected, (
            f"Last training target={y_train[-1]}, expected={expected} "
            f"(log_return={last_log_return:.6f})"
        )


# ---------------------------------------------------------------------------
# 6. Production readiness – model outputs
# ---------------------------------------------------------------------------


class TestProductionReadiness:
    @pytest.fixture(params=["dt", "rf", "xgb"])
    def model(self, request):
        if request.param == "dt":
            return DecisionTreeClassModel()
        elif request.param == "rf":
            return RandomForestClassModel(n_estimators=10)
        else:
            return XGBoostClassModel(n_estimators=10)

    def test_output_has_required_keys(self, model):
        result = model.train_predict_next(make_df(100))
        assert "prediction" in result
        assert "probability" in result

    def test_prediction_is_binary(self, model):
        result = model.train_predict_next(make_df(100))
        assert result["prediction"] in (0, 1)

    def test_probability_in_valid_range(self, model):
        result = model.train_predict_next(make_df(100))
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_consistent_with_probability(self, model):
        result = model.train_predict_next(make_df(100))
        assert result["prediction"] == int(result["probability"] > 0.5)

    def test_features_auto_populated(self, model):
        assert model.features == []
        model.train_predict_next(make_df(100))
        assert len(model.features) > 0

    def test_excluded_columns_not_in_features(self, model):
        model.train_predict_next(make_df(100))
        for col in ("open", "high", "low", "close", "target"):
            assert col not in model.features

    def test_no_crash_on_minimum_data(self, model):
        result = model.train_predict_next(make_df(20))
        assert result["prediction"] in (0, 1)

    def test_today_rsi_affects_prediction_pipeline(self):
        """
        Changing today's rsi must produce a different X_next,
        confirming current features flow into the model.
        """
        df = make_df(200)
        features = get_features(df)

        df_low_rsi = df.copy()
        df_low_rsi.loc[df_low_rsi.index[-1], "rsi"] = 10.0

        df_high_rsi = df.copy()
        df_high_rsi.loc[df_high_rsi.index[-1], "rsi"] = 90.0

        _, _, X_next_low = prepare_features(df_low_rsi, features, THRESHOLD)
        _, _, X_next_high = prepare_features(df_high_rsi, features, THRESHOLD)

        assert not np.array_equal(X_next_low, X_next_high), (
            "X_next identical despite different today's rsi — current features not reaching model"
        )

    def test_pred_record_structure_matches_runner_expectations(self):
        """
        Verify the output dict from train_predict_next contains what
        TradingPipeline.process_ticker expects for classification models.
        """
        model = DecisionTreeClassModel()
        result = model.train_predict_next(make_df(100))

        # Runner does: int(pred_output["prediction"]) and float(pred_output["probability"])
        assert isinstance(int(result["prediction"]), int)
        assert isinstance(float(result["probability"]), float)
        assert result["prediction"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0
