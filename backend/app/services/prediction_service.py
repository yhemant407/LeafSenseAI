from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class PredictionService:

    IMAGE_SIZE = (224, 224)

    CLASS_NAMES = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

    @staticmethod
    def _load_model():

        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "best_plant_disease_model.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        return tf.keras.models.load_model(model_path)

    @staticmethod
    async def predict(file):

        # Load image from uploaded file
        image = Image.open(file.file)

        # Convert to RGB
        image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(
            PredictionService.IMAGE_SIZE
        )

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32,
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # MobileNetV2 preprocessing
        image_array = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
        )

        # Load model
        model = PredictionService._load_model()

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0,
        )

        # Find highest probability
        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        predicted_class = (
            PredictionService.CLASS_NAMES[
                predicted_index
            ]
        )

        return {
            "prediction": predicted_class,
            "confidence": confidence,
        }