from tensorflow.keras.models import load_model
model = load_model("alzheimer_stage_classifier.keras", compile=False)
print("Input:", model.input)
print("Output:", model.output)