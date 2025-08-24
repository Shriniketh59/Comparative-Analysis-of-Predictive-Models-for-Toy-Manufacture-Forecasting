# deep_analysis.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================
# 1. Load Dataset
# ==============================
FILE_PATH = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df=pd.read_excel(FILE_PATH)
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}")


# ==============================
# 2. Preprocess
# ==============================
# Ensure consistent column names
df.columns = [col.strip() for col in df.columns]

if not {"State", "Year", "Number of Manufacturers"}.issubset(df.columns):
    raise KeyError(f"Missing required columns in file. Found: {list(df.columns)}")

# Sort and create demand trend (Increase = 1, Decrease = 0)
df = df.sort_values(by=["State", "Year"])
df["Demand_Change"] = df.groupby("State")["Number of Manufacturers"].diff().fillna(0)
df["Target"] = (df["Demand_Change"] > 0).astype(int)

X = df[["Year", "Number of Manufacturers"]]
y = df["Target"]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3. Build Deep Learning Model
# ==============================
model_file = ".venv/toy_demand_model.h5"

if os.path.exists(model_file):
    print("📂 Loading existing model...")
    model = load_model(model_file)
else:
    print("🛠 Training new model...")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  # Probability output
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=15, batch_size=16, verbose=1)

    # Save model
    model.save(model_file)
    print(f"✅ Model saved as {model_file}")

# ==============================
# 4. Predict with Probabilities
# ==============================
y_probs = model.predict(X_test).flatten()
y_pred = (y_probs > 0.5).astype(int)

results = pd.DataFrame({
    "State": df.iloc[y_test.index]["State"].values,
    "Year": df.iloc[y_test.index]["Year"].values,
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability_Increase": y_probs
})

# Save results for Tableau
results.to_csv("prediction_results.csv", index=False)
print("📊 Predictions saved to prediction_results.csv")

# ==============================
# 5. Visualization
# ==============================
# Pie chart: Probability distribution
plt.figure(figsize=(6,6))
plt.pie(
    [np.mean(y_probs), 1 - np.mean(y_probs)],
    labels=["Increase", "Decrease"],
    autopct="%.1f%%",
    startangle=90,
    colors=["green", "red"]
)
plt.title("Probability of Manufacturer Demand (Increase vs Decrease)")
plt.show()

# Area graph: Probability trend
plt.figure(figsize=(10,5))
plt.fill_between(range(len(y_probs)), y_probs, alpha=0.5, color="blue")
plt.plot(y_probs, color="blue")
plt.axhline(0.5, color="red", linestyle="--")
plt.title("Predicted Probability of Demand Increase")
plt.xlabel("Test Samples")
plt.ylabel("Probability")
plt.show()

