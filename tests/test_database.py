import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.pipeline.database import DatabaseService

@pytest.fixture
def db_service(mocker):
    """Vytvoří instanci DatabaseService s mockovaným enginem."""
    mocker.patch('src.pipeline.database.create_engine')
    mocker.patch('src.pipeline.database.MetaData')
    service = DatabaseService()
    service.engine = MagicMock()
    service.metadata = MagicMock()
    return service

def test_save_market_data_index_handling(db_service):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [95.0, 96.0],
        "close": [102.0, 103.0],
        "volume": [1000, 1100]
    }).set_index("date")
    
    mock_table = MagicMock()
    db_service.metadata.tables = {"market_data": mock_table}

    db_service.save_market_data(df, "AAPL")
    
    assert db_service.engine.begin.called

def test_save_prediction_calls_engine(db_service):
    mock_table = MagicMock()
    db_service.metadata.tables = {"predictions_classification": mock_table}
    
    record = {
        "ticker": "AAPL", "model": "Test", "prediction_date": "2023-01-01",
        "target_date": "2023-01-02", "predicted_class": 1, "probability": 0.6
    }
    
    db_service.save_prediction(record, "classification")
    assert db_service.engine.begin.called