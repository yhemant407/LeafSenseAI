from fastapi import HTTPException, UploadFile

from app.ml.preprocess import ImagePreprocessor
from app.schemas.prediction import PredictionResponse
from app.ml.predictor import ModelPredictor

class PredictionService:

    @staticmethod
    async def predict(file: UploadFile) -> PredictionResponse:

        # Validate uploaded file
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed."
            )

        # Preprocess image
        processed_image = await ImagePreprocessor.preprocess(file)

        print("=" * 50)
        print("Image preprocessing successful")
        print(f"Shape: {processed_image.shape}")
        print(f"Data Type: {processed_image.dtype}")
        print(f"Min Value: {processed_image.min()}")
        print(f"Max Value: {processed_image.max()}")
        print("=" * 50)

        prediction, confidence = ModelPredictor.predict(
            processed_image
        )

        return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )