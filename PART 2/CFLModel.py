import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "draft.csv")
df = pd.read_csv(csv_path)

# Split into train/test (~15% test)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df.to_csv("draft_train.csv", index=False)
test_df.to_csv("draft_test.csv", index=False)

# Reload train and test sets
train_df = pd.read_csv("draft_train.csv")
test_df = pd.read_csv("draft_test.csv")

# Separate labels
y_train = train_df["drafted"].astype("float32")
y_test = test_df["drafted"].astype("float32")

# Drop label column from features
X_train = train_df.drop("drafted", axis=1)
X_test = test_df.drop("drafted", axis=1)

# One-hot encode the categorical column
X_train = pd.get_dummies(X_train, columns=["position"])
X_test = pd.get_dummies(X_test, columns=["position"])

# Ensure both sets have exact same columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert to float32 tensors
X_train = X_train.astype("float32").values
X_test = X_test.astype("float32").values

# Set up the model(basic)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=2
)

# Get predictions
y_train_pred = (tf.sigmoid(model.predict(X_train)) > 0.5).numpy().astype(int)
y_test_pred = (tf.sigmoid(model.predict(X_test)) > 0.5).numpy().astype(int)

# Training accuracy
train_acc = np.mean(y_train_pred.flatten() == y_train)
# Test accuracy
test_acc = np.mean(y_test_pred.flatten() == y_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Test Data):")
print(cm)

# Accuracies
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Classification report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))