from flask import Flask, request, render_template, redirect, url_for
from analysis import analyzer
import os
import base64

# Initialize the Flask application
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Ensure an 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    # Render the main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if analyzer is None:
        return render_template('index.html', error="Model is not loaded. Please check server logs.")

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image_bytes = file.read()

        try:
            result = analyzer.predict(image_bytes)
            # Also send the original image back to display it
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return render_template('index.html', prediction=result, image_data=image_base64)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return render_template('index.html', error="Failed to analyze the image.")

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)