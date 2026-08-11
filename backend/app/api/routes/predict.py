from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)):
    return await PredictionService.predict(file)