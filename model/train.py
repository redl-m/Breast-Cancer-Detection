from model import create_model
from datasets import get_datasets
import os

# --- Configuration ---
EPOCHS = 15
MODEL_SAVE_PATH = '../saved_model/breast_cancer_model.keras'

# --- Main Training Script ---
if __name__ == '__main__':
    # 1. Load Data, the function also returns a test set
    train_dataset, val_dataset, test_dataset, inferred_class_names = get_datasets()

    # 2. Define User-Friendly Class Names
    class_names_map = {
        '0': 'Benign',
        '1': 'Malignant'
    }

    user_friendly_class_names = [class_names_map[name] for name in inferred_class_names]

    print(f"\nMapping inferred labels {inferred_class_names} to {user_friendly_class_names}")

    # 3. Create Model
    model = create_model(input_shape=(224, 224, 3))
    print("\nModel Summary:")
    model.summary()

    # 4. Train the Model
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )
    print("Model training finished.")

    # 5. Evaluate the Model on the Test Set
    print("\nEvaluating model on the test set...")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Save the Model and Class Names
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")

    with open('../saved_model/class_names.txt', 'w') as f:
        for name in user_friendly_class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to ../saved_model/class_names.txt")