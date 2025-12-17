# Breast-Cancer-Detection

## About

This project is a web-based application built for the detection and analysis of breast cancer patterns in medical imagery. It uses a Deep Learning model trained on [kaggle's Breast Cancer dataset shared by "HAYDER ."](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) using [TensorFlow](https://www.tensorflow.org) version 2.20.0 and [Keras](https://keras.io) version 3.11.2.

The system performs binary classification to predict the class of the uploaded image (e.g., Benign vs. Malignant) and calculates a confidence score. It utilizes **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the regions of the image that led to the model's prediction. The image processing and heatmap superimposition are handled using [NumPy](https://numpy.org) version 1.26.4, [Pillow](https://python-pillow.org) version 10.3.0, [OpenCV](https://opencv.org) version 4.12.0 and [Matplotlib](https://matplotlib.org) version 3.8.4.

The backend is powered by [Flask](https://flask.palletsprojects.com) version 3.1.1.

## Getting Started

To run a local copy of the project, please follow the instructions below.

### Data

datasets.py expects to find three folders with test, training and validation data to be found in: root/data, which can be received from [kaggle's Breast Cancer dataset shared by "HAYDER ."](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection).

### Prerequisites & Model Setup

The application requires a pre-trained Keras model and a class names file to function. Ensure the directory structure is set up as follows:

* **Root Directory**
    * `server/` (Contains `app.py` and `analysis.py`)
    * `saved_model/`
        * `breast_cancer_model.keras` (Trained TensorFlow model)
        * `class_names.txt` (Text file containing class names, one per line)

### Development server

To start a local development server, navigate to the server directory:

```bash
cd .\server\
```
and run:

```bash
python app.py
```

Once the server is running, open your browser and navigate to `http://localhost:5000/`.

## Program Usage

### Basics

To analyze an image, navigate to the homepage and use the upload interface to select a medical image file. Once uploaded, the system will process the image using the loaded Deep Learning model.

### Analysis & Visualization

After processing, the application displays:
1.  **Prediction:** The predicted class (based on the classes defined in `class_names.txt`).
2.  **Confidence:** The probability score associated with the prediction.
3.  **Visual Explanation (Grad-CAM):** The original image with a superimposed heatmap. Red/warm areas indicate regions of high importance that heavily influenced the model's decision, allowing for visual verification of the diagnosis logic.

### Configuration

The model configuration is handled within `analysis.py`. The `ImageAnalyzer` class is initialized with default paths in `analysis.py`:

```python
analyzer = ImageAnalyzer(
    model_path='../saved_model/breast_cancer_model.keras',
    class_names_path='../saved_model/class_names.txt'
)
```
## License

Distributed under CC-0. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Michael Redl - [Personal Website](https://michaeljosefredl.at) - [@redl_m](https://www.instagram.com/redl__m/) - michael.redl14042004@gmail.com

Project Link: [https://github.com/redl-m/Breast-Cancer-Detection](https://github.com/redl-m/Breast-Cancer-Detection)
