from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(
        ge=0,
        le=100,
        description="Prediction confidence percentage"
    )