import pandas as pd
from unittest.mock import MagicMock
from src.pipeline.runner import TradingPipeline

def test_evaluate_prediction_logic():
    db_mock = MagicMock()
    pipeline = TradingPipeline(db_mock)
    
    ticker = "AAPL"
    model = MagicMock(name="TestModel", model_type="classification", classification_threshold=0.005)
    model.name = "TestModel"
    
    target_date = "2023-01-02"
    df_actual = pd.DataFrame({"log_return": [0.01]}, index=pd.to_datetime([target_date]))
    
    db_mock.get_prediction_for_evaluation.return_value = {"id": 1, "predicted_value": 1}
    
    res = pipeline.evaluate_prediction(ticker, df_actual, model, target_date)
    
    assert res is not None
    assert res["correct"] is True
    assert res["actual_class"] == 1