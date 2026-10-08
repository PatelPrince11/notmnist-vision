# CFLModel.py
# Original Author: Prince Patel
# Semister: Fall 2025
# CPSC 433 L01 - T05

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "draft.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Split train/test (~15% test)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df.to_csv("draft_train.csv", index=False)
test_df.to_csv("draft_test.csv", index=False)

# Prepare features and labels
X_train = train_df.drop("drafted", axis=1)
y_train = train_df["drafted"]
X_test = test_df.drop("drafted", axis=1)
y_test = test_df["drafted"]

# One-hot encode categorical column
X_train = pd.get_dummies(X_train, columns=["position"])
X_test = pd.get_dummies(X_test, columns=["position"])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.values.astype("float32")
y_test = y_test.values.astype("float32")


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=2
)

# Predict probabilities and convert to class labels
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Test Data):")
print(cm)

# Classification report
cr = classification_report(y_test, y_pred)
print("\nClassification Report (Test Data):")
print(cr)

# Training vs Test Accuracy
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")