from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from load_data import test_generator

# Load the trained model
model = load_model("alzheimer_stage_classifier.h5")

# Predict on test data
predictions = model.predict(test_generator, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

accuracy = np.mean(predicted_classes == true_classes)
print(f"Overall Accuracy: {accuracy:.2%}")

# Print evaluation metrics
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix heatmap
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()