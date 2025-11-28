from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Sample data
predicted_stage = "Non Demented"
confidence_scores = {
    "Mild Dementia": 0.1770,
    "Moderate Dementia": 0.2914,
    "Non Demented": 0.4594,
    "Very mild Dementia": 0.0839
}

image_path = "static/uploaded_image.png"
chart_path = "static/confidence_chart.png"

# Generate chart
labels = list(confidence_scores.keys())
scores = [score * 100 for score in confidence_scores.values()]
plt.figure(figsize=(6, 4))
plt.bar(labels, scores, color='skyblue')
plt.ylim(0, 100)
plt.title("Confidence Scores")
plt.ylabel("Confidence (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(chart_path, format='png')
plt.close()

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Alzheimer Stage Detection Report", ln=True, align='C')
pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

pdf.ln(10)
pdf.cell(200, 10, txt=f"Predicted Stage: {predicted_stage}", ln=True)

pdf.ln(5)
pdf.cell(200, 10, txt="Confidence Scores:", ln=True)
for label, score in confidence_scores.items():
    pdf.cell(200, 10, txt=f"{label}: {score * 100:.2f}%", ln=True)

# Embed MRI image on a new page
if os.path.exists(image_path):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Uploaded MRI Image:", ln=True)
    try:
        pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
    except RuntimeError as e:
        pdf.cell(200, 10, txt=f"Error loading image: {str(e)}", ln=True)

# Embed confidence chart
if os.path.exists(chart_path):
    pdf.ln(10)
    pdf.cell(200, 10, txt="Confidence Chart:", ln=True)
    try:
        pdf.image(chart_path, x=10, y=pdf.get_y(), w=100)
    except RuntimeError as e:
        pdf.cell(200, 10, txt=f"Error loading chart: {str(e)}", ln=True)

# Save PDF
pdf.output("alzheimer_report.pdf")