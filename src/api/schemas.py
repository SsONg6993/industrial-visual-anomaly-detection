from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    coreset_size: int


class PredictionResponse(BaseModel):
    prediction: str = Field(
        pattern="^(good|defect)$"
    )

    anomaly_score: float
    threshold: float
    latency_ms: float