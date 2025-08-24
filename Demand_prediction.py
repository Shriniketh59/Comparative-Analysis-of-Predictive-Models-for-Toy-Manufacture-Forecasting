import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Load Dataset ----------------
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)

# Rename columns
df.rename(columns={"Number of Manufacturers": "Manufacturers"}, inplace=True)

# Sort and create lag feature
df = df.sort_values(by=["State", "Year"])
df["Prev_Manufacturers"] = df.groupby("State")["Manufacturers"].shift(1)
df = df.dropna()

# Target: Increase (1) or Decrease/No Change (0)
df["Demand_Increase"] = (df["Manufacturers"] > df["Prev_Manufacturers"]).astype(int)

# Features and Target
X = df[["Year", "Prev_Manufacturers"]]
y = df["Demand_Increase"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Deep Learning Model ----------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),   # 2 features
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")      # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# ---------------- Evaluation ----------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nDeep Learning Results:")
print(classification_report(y_test, y_pred, zero_division=0))   # avoid warnings
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- Visualization ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Training curve
ax[0].plot(history.history["accuracy"], label="Train Accuracy", color="blue")
ax[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="orange")
ax[0].set_title("Model Training Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend()

ax[1].plot(history.history["loss"], label="Train Loss", color="blue")
ax[1].plot(history.history["val_loss"], label="Val Loss", color="orange")
ax[1].set_title("Model Training Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()

plt.show()

# ---------------- Prediction Distribution ----------------
def plot_prediction_distribution(preds, title):
    labels_map = {0: "Increases", 1: "Increase"}
    counts = pd.Series(preds.flatten()).value_counts().sort_index()
    labels = [labels_map[i] for i in counts.index]  # only existing labels
    colors = ["red" if i == 0 else "green" for i in counts.index]

    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title(title)
    plt.show()

# Overall prediction distribution
plot_prediction_distribution(y_pred, "Deep Learning Demand Prediction Distribution")
