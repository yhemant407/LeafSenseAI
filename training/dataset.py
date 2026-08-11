from config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

import tensorflow as tf


def get_class_names():
    return sorted(
        [
            folder.name
            for folder in TRAIN_DIR.iterdir()
            if folder.is_dir()
        ]
    )


def check_dataset():

    for directory in (TRAIN_DIR, VALID_DIR, TEST_DIR):
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

    classes = get_class_names()

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Train Directory      : {TRAIN_DIR}")
    print(f"Validation Directory : {VALID_DIR}")
    print(f"Test Directory       : {TEST_DIR}")

    print(f"\nNumber of Classes: {len(classes)}")

    return classes


def load_datasets():

    class_names = get_class_names()

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .cache()
        .prefetch(AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .cache()
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .cache()
        .prefetch(AUTOTUNE)
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    )


def inspect_dataset(dataset, class_names):

    print("\nInspecting Dataset...\n")

    print("Class Names:")
    print(class_names)

    for images, labels in dataset.take(1):

        print(f"\nImage Batch Shape : {images.shape}")
        print(f"Label Batch Shape : {labels.shape}")
        print(f"Image Data Type   : {images.dtype}")
        print(f"Label Data Type   : {labels.dtype}")


if __name__ == "__main__":

    check_dataset()

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = load_datasets()

    inspect_dataset(train_dataset, class_names)