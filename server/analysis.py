import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2  # OpenCV for image manipulation
import base64
import matplotlib.cm as cm


class ImageAnalyzer:
    def __init__(self, model_path, class_names_path):
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded class names: {self.class_names}")

        # Find the last convolutional layer for Grand-CAM
        self.last_conv_layer_name = self._find_last_conv_layer()
        print(f"Found last conv layer for Grad-CAM: {self.last_conv_layer_name}")

        inputs = tf.keras.Input(shape=(224, 224, 3), name="gradcam_input")
        outputs = self.model(inputs, training=False)

        # TODO: Fix Grand-CAM heatmap error
        """
        Could not generate Grad-CAM heatmap: "Exception encountered when calling Functional.call().\n\n\x1b[1m1938658692368\x1b[0m\n\nArgument
        s received by Functional.call():\n  • inputs=tf.Tensor(shape=(1, 224, 224, 3), dtype=float32)\n  • training=None\n  • mask=None\n  • kwargs=<class 'inspect._empty'>"
        """
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )

        print("Model built successfully.")

    def _find_last_conv_layer(self):
        """
        Finds the name of the last Conv2D layer by checking the layer's type.
        This is the most reliable method and works for both simple and nested models.
        """
        # Iterate through the model's layers in reverse
        for layer in reversed(self.model.layers):
            # If a layer is a model itself, search inside it
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return sub_layer.name

            # Check if the top-level layer is a Conv2D layer
            elif isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("Could not find a Conv2D layer in the model for Grad-CAM.")

    def _make_gradcam_heatmap(self, img_array):
        """Generates a Grad-CAM heatmap."""
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization, we will normalize the heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max == 0:
            heatmap_max = 1e-10  # Add a small constant to prevent division by zero
        heatmap = heatmap / heatmap_max
        return heatmap.numpy()

    def _superimpose_gradcam(self, img_bytes, heatmap, alpha=0.4):
        """Superimposes the heatmap on the original image."""
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
        img = cv2.resize(img, (224, 224))

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We use a colormap to colorize the heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[np.uint8(255 * heatmap)]

        # Convert to BGR for OpenCV
        jet_heatmap = cv2.cvtColor(np.float32(jet_heatmap), cv2.COLOR_RGB2BGR)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + cv2.cvtColor(np.float32(img / 255.0), cv2.COLOR_RGB2BGR)
        superimposed_img = np.clip(superimposed_img, 0, 1)

        # Convert back to uint8
        superimposed_img = np.uint8(255 * superimposed_img)

        # Encode to JPEG bytes
        is_success, buffer = cv2.imencode(".jpg", superimposed_img)
        if not is_success:
            raise ValueError("Could not encode heatmap image.")

        return buffer.tobytes()

    def preprocess_image(self, image_bytes):
        """Takes image bytes, preprocesses it to the model's required format."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    def predict(self, image_bytes):
        """Analyzes an image and returns the prediction, confidence, and heatmap."""
        processed_image = self.preprocess_image(image_bytes)

        # Make prediction
        prediction = self.model.predict(processed_image)
        score = prediction[0][0]

        print("INFO: Score: " + str(score))

        if score < 0.5:
            predicted_class = self.class_names[0]  # Benign
            confidence = 1 - score
        else:
            predicted_class = self.class_names[1]  # Malignant
            confidence = score

        # Generate and apply Grad-CAM heatmap
        try:
            heatmap = self._make_gradcam_heatmap(processed_image)
            heatmap_bytes = self._superimpose_gradcam(image_bytes, heatmap)
            heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        except Exception as e:
            print(f"Could not generate Grad-CAM heatmap: {e}")
            heatmap_base64 = None

        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2%}",
            "raw_confidence": confidence,  # for the chart
            "heatmap_data": heatmap_base64
        }


try:
    analyzer = ImageAnalyzer(
        model_path='../saved_model/breast_cancer_model.keras',
        class_names_path='../saved_model/class_names.txt'
    )
except Exception as e:
    print(f"Error initializing ImageAnalyzer: {e}")
    analyzer = None
