import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# 1. Load data
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)

# Compute YoY change
df = df.sort_values(by=["State", "Year"])
df["YoY_Change_%"] = df.groupby("State")["Number of Manufacturers"].pct_change() * 100
df = df.dropna()

# 2. Prepare sequences
sequence_length = 5
X, y = [], []
for state in df["State"].unique():
    changes = df[df["State"] == state]["YoY_Change_%"].values
    for i in range(len(changes) - sequence_length):
        seq = changes[i:i+sequence_length]
        label = 1 if np.std(seq) > 10 else 0   # Volatile if std > 10
        X.append(seq)
        y.append(label)

X = np.array(X).reshape(len(X), sequence_length, 1)
y = np.array(y)

# Train-Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build model
model = Sequential([
    LSTM(50, input_shape=(sequence_length, 1)),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 4. Train with history
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20, batch_size=8, verbose=1)

# 5. Save
model.save("C:/Users/revat/PycharmProjects/Toy manufracture analysis in tableau ML Model/volatility_model.keras")
print("✅ Model trained and saved successfully!")

# 6. Plot training history
plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
