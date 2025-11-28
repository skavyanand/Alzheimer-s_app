import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

# Load model
MODEL_PATH = "alzheimer_stage_classifier.h5"
model = load_model(MODEL_PATH, compile=False)

# Diagnostic check
try:
    print("Model input:", model.input)
    print("Model output:", model.output)
except Exception as e:
    print("Model input/output not available:", e)

# Class labels
class_labels = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 1)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence_scores = {label: float(f"{pred:.4f}") for label, pred in zip(class_labels, predictions)}

    return jsonify({
        'predicted_stage': predicted_label,
        'confidence_scores': confidence_scores
    })

if __name__ == '__main__':
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    app.run(debug=True)