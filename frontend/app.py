import os
import io
import traceback
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_file
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'alzheimer_stage_classifier.h5')
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print("❌ ERROR loading model:")
    traceback.print_exc()
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if image:
        image_path = os.path.join(STATIC_DIR, 'uploaded_image.png')

        # Save image for display
        img_display = Image.open(image).convert('RGB')
        img_display.save(image_path, format='PNG')

        # Preprocess for model
        img = Image.open(image).convert('L')  # Grayscale
        img = img.resize((128, 128))  # Match model input
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 1)  # Shape: (1, 128, 128, 1)

        # Pad or reshape to match model input if needed
        expected_shape = model.input_shape[1]
        if img_array.shape[1] != expected_shape:
            print(f"⚠️ Adjusting input shape from {img_array.shape[1]} to {expected_shape}")
            if img_array.shape[1] < expected_shape:
                # Pad with zeros
                pad_width = expected_shape - img_array.shape[1]
                img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Trim excess
                img_array = img_array[:, :expected_shape]

        # Define labels globally for this route
        labels = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

        # Predict
        if model:
            try:
                predictions = model.predict(img_array)[0]
                confidence_scores = dict(zip(labels, predictions.tolist()))
                predicted_stage = labels[np.argmax(predictions)]
            except Exception as e:
                print("❌ Prediction error:")
                traceback.print_exc()
                predicted_stage = "Model Error"
                confidence_scores = {label: 0.0 for label in labels}
        else:
            predicted_stage = "Model Error"
            confidence_scores = {label: 0.0 for label in labels}

        result = {
            "predicted_stage": predicted_stage,
            "confidence_scores": confidence_scores,
            "image_path": image_path
        }

        return render_template('index.html', result=result)

    return redirect(url_for('index'))

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    result = request.form

    predicted_stage = result['predicted_stage']
    labels = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']
    scores = [float(result.get(label, 0)) for label in labels]

    chart_path = os.path.join(STATIC_DIR, 'confidence_chart.png')
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [s * 100 for s in scores], color='skyblue')
    plt.ylim(0, 100)
    plt.title("Confidence Scores")
    plt.ylabel("Confidence (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_path, format='png')
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Alzheimer Stage Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {predicted_stage}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Confidence Scores:", ln=True)
    for label, score in zip(labels, scores):
        pdf.cell(200, 10, txt=f"{label}: {score * 100:.2f}%", ln=True)

    image_path = result.get('image_path')
    if image_path and os.path.exists(image_path):
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Uploaded MRI Image:", ln=True)
        try:
            y_position = pdf.get_y() + 5
            pdf.image(image_path, x=10, y=y_position, w=180)
        except RuntimeError as e:
            pdf.cell(200, 10, txt=f"Error loading image: {str(e)}", ln=True)

    if os.path.exists(chart_path):
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Confidence Chart:", ln=True)
        try:
            y_position = pdf.get_y() + 5
            pdf.image(chart_path, x=10, y=y_position, w=180)
        except RuntimeError as e:
            pdf.cell(200, 10, txt=f"Error loading chart: {str(e)}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_stream = io.BytesIO(pdf_bytes)
    return send_file(pdf_stream, download_name="alzheimer_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)