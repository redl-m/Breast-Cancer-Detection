import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import base64
import matplotlib.cm as cm

def _is_conv(layer):
    return isinstance(layer, tf.keras.layers.Conv2D)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img)

    arr = arr / 255.0 # normalized values

    arr = tf.expand_dims(arr, 0)
    return arr


def _superimpose_gradcam(img_bytes, heatmap, alpha=0.4):
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap)]
    jet_heatmap = cv2.cvtColor(np.float32(jet_heatmap), cv2.COLOR_RGB2BGR)

    superimposed = jet_heatmap * alpha + cv2.cvtColor(np.float32(img / 255.0), cv2.COLOR_RGB2BGR)
    superimposed = np.clip(superimposed, 0, 1)
    superimposed = np.uint8(255 * superimposed)

    ok, buf = cv2.imencode(".jpg", superimposed)
    if not ok:
        raise ValueError("Could not encode heatmap image.")
    return buf.tobytes()


class ImageAnalyzer:
    def __init__(self, model_path, class_names_path, min_spatial=7):
        print(f"Loading model from {model_path}...")
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded class names: {self.class_names}")

        self.model = loaded_model
        self.min_spatial = min_spatial

        # ---- Rebuild a clean graph manually, recording conv feature maps ----
        new_input = tf.keras.Input(shape=(224, 224, 3))
        x = new_input

        conv_outputs = []  # [(layer_name, tensor)]
        for layer in loaded_model.layers:
            x = layer(x)
            if _is_conv(layer):
                conv_outputs.append((layer.name, x))

        model_output = x  # final output of the rebuilt graph

        # Pick the deepest conv with spatial size >= min_spatial (fallback to last conv)
        chosen_name, chosen_tensor = None, None
        for name, tensor in reversed(conv_outputs):
            shp = tensor.shape  # (None, H, W, C)
            h, w = int(shp[1]) if shp[1] is not None else 0, int(shp[2]) if shp[2] is not None else 0
            if h >= self.min_spatial and w >= self.min_spatial:
                chosen_name, chosen_tensor = name, tensor
                break
        if chosen_tensor is None and conv_outputs:
            chosen_name, chosen_tensor = conv_outputs[-1]  # fallback
        if chosen_tensor is None:
            raise ValueError("Could not find a Conv2D layer to use for Grad-CAM.")

        self.last_conv_layer_name = chosen_name
        print(f"Grad-CAM will use conv layer: {self.last_conv_layer_name} (spatial {chosen_tensor.shape[1]}×{chosen_tensor.shape[2]})")

        # Grad model built from the manually re-applied layers
        self.grad_model = tf.keras.models.Model(inputs=new_input, outputs=[chosen_tensor, model_output])
        print("Robust Grad-CAM model built successfully.")

    def _make_gradcam_heatmap(self, img_array):
        """Generates a Grad-CAM heatmap (handles binary logits & multi-class)."""

        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_array, training=False)

            if preds.shape[-1] == 1:
                p = tf.clip_by_value(preds, 1e-7, 1 - 1e-7)
                logits = tf.math.log(p) - tf.math.log(1.0 - p)  # logit(p)
                class_channel = logits[:, 0]
            else:
                # If multi-class, use the predicted class score
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise RuntimeError("Gradients are None. The graph may be disconnected.")

        # Channel-wise weights (GAP over spatial dims)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]  # [H, W, C]

        # Weighted sum over channels -> [H, W]
        heatmap = conv_out @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize to [0,1]
        heatmap = tf.nn.relu(heatmap)
        maxv = tf.reduce_max(heatmap)
        heatmap = heatmap / (maxv + 1e-8)
        return heatmap.numpy()

    def predict(self, image_bytes):
        processed = preprocess_image(image_bytes)

        # Prediction for class/score readout
        prediction = self.model(processed, training=False).numpy()
        score = float(prediction[0][0])

        if score < 0.5:
            predicted_class = self.class_names[0]
            confidence = 1 - score
        else:
            predicted_class = self.class_names[1]
            confidence = score

        # Grad-CAM
        try:
            heatmap = self._make_gradcam_heatmap(processed)
            heatmap_bytes = _superimpose_gradcam(image_bytes, heatmap)
            heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        except Exception as e:
            print(f"Could not generate Grad-CAM heatmap: {e}")
            heatmap_base64 = None

        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2%}",
            "raw_confidence": confidence,
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