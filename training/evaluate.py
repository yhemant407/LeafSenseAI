import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from config import (
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
)

from dataset import load_datasets


def evaluate():

    print("\nLoading Best Model...\n")

    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    _, _, test_dataset, class_names = load_datasets()

    print("\nEvaluating on Test Dataset...\n")

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    print(f"\nTest Accuracy : {accuracy:.4f}")
    print(f"Test Loss     : {loss:.4f}")

    print("\nGenerating Predictions...\n")

    predictions = model.predict(
        test_dataset,
        verbose=1,
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1,
    )

    true_labels = np.concatenate(
        [
            labels.numpy()
            for _, labels in test_dataset
        ]
    )

    # ==========================
    # Classification Report
    # ==========================

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
    )

    print("\nClassification Report\n")
    print(report)

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
    ) as file:
        file.write(report)

    print(
        f"\nClassification report saved to:\n"
        f"{CLASSIFICATION_REPORT_PATH}"
    )

    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    plt.figure(figsize=(16, 14))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_PATH}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    evaluate()