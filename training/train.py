from dataset import (
    check_dataset,
    load_datasets,
)

from model import build_model
import tensorflow as tf

from config import (
    EPOCHS,
    MODEL_PATH,
    BEST_MODEL_PATH,
    PATIENCE
)
from utils import save_training_history


def main():

    # Verify dataset
    classes = check_dataset()

    # Load datasets
    train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

    # Build model
    model = build_model(
        num_classes=len(classes)
    )

    # Display model architecture
    model.summary()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting Training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
    )
    save_training_history(history)

    print("\nTraining Complete!")

    model.save(MODEL_PATH)

    print(f"\nModel saved to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()