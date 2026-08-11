import tensorflow as tf

from config import IMAGE_SIZE
from config import (
    IMAGE_SIZE,
    LEARNING_RATE,
    DROPOUT_RATE,
)

def build_model(num_classes: int):

    # Load MobileNetV2 without its original classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build classifier
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # Preprocess exactly as MobileNetV2 expects
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Feature extraction
    x = base_model(x, training=False)

    # Reduce feature maps
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Prevent overfitting
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    # Final classifier
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model