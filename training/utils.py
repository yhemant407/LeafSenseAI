import matplotlib.pyplot as plt

from config import (
    ACCURACY_PLOT_PATH,
    LOSS_PLOT_PATH,
)


def save_training_history(history):

    accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)

    # ----------------------------
    # Accuracy Plot
    # ----------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        accuracy,
        label="Training Accuracy",
        linewidth=2,
    )

    plt.plot(
        epochs,
        validation_accuracy,
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT_PATH,
        dpi=300,
    )

    plt.close()

    # ----------------------------
    # Loss Plot
    # ----------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        loss,
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        validation_loss,
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT_PATH,
        dpi=300,
    )

    plt.close()

    print("\nTraining history saved successfully.")