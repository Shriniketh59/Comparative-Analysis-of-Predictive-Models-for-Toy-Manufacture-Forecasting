import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- Load Dataset ----------------
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)

# Rename for consistency
df.rename(columns={"Number of Manufacturers": "Manufacturers"}, inplace=True)

# Sort by State & Year
df = df.sort_values(by=["State", "Year"])

# YoY % Change
df["YoY_Change_%"] = df.groupby("State")["Manufacturers"].pct_change() * 100
df = df.dropna()

# ---------------- Volatility Label ----------------
# Compute volatility per state
volatility = df.groupby("State")["YoY_Change_%"].std().reset_index()
volatility.columns = ["State", "Volatility"]

# Define stable vs volatile
threshold = volatility["Volatility"].median()
volatility["Label"] = np.where(volatility["Volatility"] > threshold, "Volatile", "Stable")

# Merge back with df
df = df.merge(volatility[["State", "Label"]], on="State", how="left")

# ---------------- Prepare Sequence Data ----------------
sequence_length = 3  # use last 3 years as input
X, y = [], []

states = df["State"].unique()

for state in states:
    state_data = df[df["State"] == state].sort_values("Year")
    changes = state_data["YoY_Change_%"].values
    label = state_data["Label"].iloc[0]  # whole state has one label

    for i in range(len(changes) - sequence_length):
        X.append(changes[i:i + sequence_length])
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)  # Stable=0, Volatile=1

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- Build Deep Learning Model ----------------
model = models.Sequential([
    layers.LSTM(64, input_shape=(sequence_length, 1), return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# ---------------- Evaluation ----------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nDeep Learning Classification Results:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- Prediction Distribution ----------------
pred_counts = pd.Series(y_pred.flatten()).value_counts()
plt.pie(pred_counts, labels=le.classes_, autopct="%1.1f%%", colors=["green", "red"])
plt.title("Predicted Stable vs Volatile States")
plt.show()
