import numpy as np
import tensorflow as tf

# Load test data
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_test = f['x_test']
    y_test = f['y_test']

x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)  # add channel dimension

# Indices to check
indices = [21, 102, 261, 615, 918, 1201, 3233]

# Load models
partial_model = tf.keras.models.load_model("notMNIST-Partial.keras")
complete_model = tf.keras.models.load_model("notMNIST-Complete.keras")

# Function to print prediction results
def print_predictions(model, model_name):
    print(f"\n=== Predictions using {model_name} ===")
    for idx in indices:
        img = x_test[idx].reshape(1, 28, 28, 1)
        true_label = y_test[idx]
        pred = model.predict(img, verbose=0)[0]
        predicted_label = np.argmax(pred)
        percentages = [f"{p*100:.2f}%" for p in pred]
        print(f"\nIndex: {idx}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence (% per class): {percentages}")

# Run predictions for Partial model
print_predictions(partial_model, "notMNIST-Partial.keras")

# Run predictions for Complete model
print_predictions(complete_model, "notMNIST-Complete.keras")