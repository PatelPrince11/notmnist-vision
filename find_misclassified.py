import tensorflow as tf
import numpy as np

# Load dataset
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)   # shape (10000, 28, 28, 1)

# Load your partial model
model = tf.keras.models.load_model("notMNIST-Partial.keras")

# Get predictions
pred_prob = model.predict(x_test, verbose=0)
pred_labels = np.argmax(pred_prob, axis=1)

# Find misclassified indices
misclassified = np.where(pred_labels != y_test)[0]

print(f"Found {len(misclassified)} misclassified images.")
print("Here are the first 30 (index, true_label, predicted_label):\n")

for idx in misclassified[:30]:
    print(f"Index {idx}  True: {y_test[idx]}  Predicted: {pred_labels[idx]}")