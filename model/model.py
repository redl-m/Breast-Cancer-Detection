from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3)):
    """
    Creates and compiles a CNN model for binary classification.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps to a 1D vector
        layers.Flatten(),

        # Dense (fully-connected) layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization to prevent overfitting

        # Output layer, 'sigmoid' activation outputs a probability between 0 and 1.
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':

    model = create_model()
    model.summary()