from src.api.schemas import HealthResponse, PredictionResponse


def test_health_response_schema():
    result = HealthResponse(
        status="ok",
        device="cuda",
        model_loaded=True,
        coreset_size=72979,
    )

    assert result.status == "ok"
    assert result.device == "cuda"
    assert result.model_loaded is True
    assert result.coreset_size == 72979


def test_prediction_response_schema():
    result = PredictionResponse(
        prediction="defect",
        anomaly_score=0.72,
        threshold=0.569758,
        latency_ms=20.4,
    )

    assert result.prediction == "defect"
    assert result.anomaly_score == 0.72
    assert result.threshold == 0.569758
    assert result.latency_ms == 20.4