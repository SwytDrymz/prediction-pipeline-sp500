import pytest
import pandas as pd
import numpy as np
from src.pipeline.collector import add_features


@pytest.fixture
def sample_df():
    """Generuje dummy data pro testy."""
    dates = pd.date_range(start="2020-01-01", periods=250)
    return pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, size=250),
            "high": np.random.uniform(110, 120, size=250),
            "low": np.random.uniform(90, 100, size=250),
            "close": np.random.uniform(100, 110, size=250),
            "volume": np.random.randint(1000, 5000, size=250),
        },
        index=dates,
    )


def test_add_features_success(sample_df):
    input_df = sample_df.reset_index().rename(columns={"index": "date"})

    df_featured = add_features(input_df)

    assert "rsi_14" in df_featured.columns
    assert "macd" in df_featured.columns
    assert "log_return" in df_featured.columns
    assert not df_featured.isnull().values.any()


def test_add_features_insufficient_data():
    df = pd.DataFrame({"close": [100, 101, 102]})
    df_featured = add_features(df)

    assert len(df_featured) == 3
    assert "rsi_14" not in df_featured.columns
