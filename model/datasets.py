import tensorflow as tf

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Update paths to point to the specific folders
TRAIN_DIR = '../data/train'
VALID_DIR = '../data/valid'
TEST_DIR = '../data/test'


def get_datasets():
    """
    Loads and preprocesses the dataset from the pre-split train, valid, and test directories.
    """
    print("Loading datasets from pre-split folders...")

    # Create the training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'  # For 0/1 labels
    )

    # Create the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Create the test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Get the class names found by the utility (will be ['0', '1'])
    class_names = train_ds.class_names
    print(f"Classes inferred from folder names: {class_names}")

    # Create a preprocessing layer for normalization
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # Apply the normalization to all datasets
    def normalize_dataset(ds):
        return ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = normalize_dataset(train_ds)
    val_ds = normalize_dataset(val_ds)
    test_ds = normalize_dataset(test_ds)

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE

    def configure_for_performance(ds):
        return ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    print("Datasets loaded and configured successfully.")
    return train_ds, val_ds, test_ds, class_names


if __name__ == '__main__':
    # You can run this file directly to test if data loading works
    train_dataset, val_dataset, test_dataset, names = get_datasets()
    print(f"\nInferred Class Names: {names}")

    # Print the shape of a single batch from the training set
    for images, labels in train_dataset.take(1):
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")