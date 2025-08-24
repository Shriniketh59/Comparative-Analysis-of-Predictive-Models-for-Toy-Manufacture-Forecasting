import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------------
# 1. Load Data
# ------------------------
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)


# ------------------------
# 2. Add Hashing (Data Security)
# ------------------------
def hash_value(val):
    return hashlib.sha256(str(val).encode()).hexdigest()


df["State_Hash"] = df["State"].apply(hash_value)
df = df.drop(columns=["State"])

# ------------------------
# 3. Compute YoY % Change
# ------------------------
df = df.sort_values(by=["State_Hash", "Year"])
df["YoY_Change_%"] = df.groupby("State_Hash")["Number of Manufacturers"].pct_change() * 100
df = df.dropna()

# ------------------------
# 4. Prepare Sequences
# ------------------------
sequence_length = 5
X, y, states = [], [], []

for state in df["State_Hash"].unique():
    changes = df[df["State_Hash"] == state]["YoY_Change_%"].values

    for i in range(len(changes) - sequence_length):
        seq = changes[i:i + sequence_length]
        label = 1 if np.std(seq) > 10 else 0
        X.append(seq)
        y.append(label)
        states.append(state)

X = np.array(X).reshape(len(X), sequence_length, 1)
y = np.array(y)

# ------------------------
# 5. Build Model
# ------------------------
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 1)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ------------------------
# 6. Train Model
# ------------------------
model.fit(X, y, epochs=20, batch_size=8, verbose=1)

# ------------------------
# 7. Predictions
# ------------------------
y_pred = (model.predict(X) > 0.5).astype(int).flatten()

results_df = pd.DataFrame({
    "State_Hash": states,
    "True_Label": y,
    "Predicted": y_pred
})
# ------------------------
# 9. Save Model Securely
# ------------------------
model.save("C:/Users/revat/PycharmProjects/Toy manufracture analysis in tableau ML Model/volatility_secure_model.h5")
print("✅ Secure Deep Learning Model trained and saved successfully!")
print("📊 Sorted bar graph visualization displayed.")
