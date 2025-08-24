import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_excel("Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx")

# Debugging: check columns
print("Columns in dataset:", df.columns)

# Correct manufacturer column
manufacturer_col = "Number of Manufacturers"

# ---------------------------
# 2. Top 10 States by Manufacturers
# ---------------------------
df_top = df.groupby("State")[manufacturer_col].sum().nlargest(10).reset_index()

# ---------------------------
# 3. Encode States
# ---------------------------
encoder = LabelEncoder()
df_top = df_top.copy()   # avoid SettingWithCopyWarning
df_top["State_Code"] = encoder.fit_transform(df_top["State"])

# ---------------------------
# 4. Prepare Features & Target
# ---------------------------
X = df_top[["State_Code"]]  # feature
y = df_top[manufacturer_col]  # target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 5. Build Deep Learning Model
# ---------------------------
model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),
    Dense(8, activation="relu"),
    Dense(1)  # Regression output
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---------------------------
# 6. Train Model
# ---------------------------
history = model.fit(X_scaled, y, epochs=20, verbose=1)

# ---------------------------
# 7. Predictions
# ---------------------------
df_top["Predicted_Manufacturers"] = model.predict(X_scaled)

# ---------------------------
# 8. Standard Deviation
# ---------------------------
std_df = df_top.groupby("State")["Predicted_Manufacturers"].std().reset_index()
print("\n📊 Standard Deviation of Predictions by State:\n", std_df)

# Bar Chart - Top 10 States
plt.figure(figsize=(10,6))
plt.bar(df_top["State"], df_top["Predicted_Manufacturers"], color="skyblue")
plt.title("Top 10 States - Predicted No. of Manufacturers (Deep Learning)")
plt.xlabel("State")
plt.ylabel("Predicted Manufacturers")
plt.xticks(rotation=20)  # small tilt for readability
plt.tight_layout()
plt.show()
