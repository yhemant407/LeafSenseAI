from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import UploadFile


class ImagePreprocessor:
    IMAGE_SIZE = (224, 224)

    @staticmethod
    async def preprocess(file: UploadFile) -> np.ndarray:
        """
        Convert uploaded image into a normalized NumPy array.
        """

        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes))

        image = image.convert("RGB")

        image = image.resize(ImagePreprocessor.IMAGE_SIZE)

        image_array = np.array(image, dtype=np.float32)

        image_array /= 255.0

        image_array = np.expand_dims(image_array, axis=0)

        return image_array