import numpy as np


class ModelPredictor:

    @staticmethod
    def predict(image: np.ndarray) -> tuple[str, float]:
        """
        Temporary predictor.

        This will later call the TensorFlow model.
        """

        # Mock prediction
        return (
            "Tomato___Healthy",
            99.72
        )