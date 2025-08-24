import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load dataset
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"  # Change to your file
df = pd.read_excel(file_path)

print("Columns in dataset:", df.columns)

# --- Step 1: Map US States to Regions ---
region_mapping = {
    # Northeast
    "Maine": "Northeast", "New Hampshire": "Northeast", "Vermont": "Northeast",
    "Massachusetts": "Northeast", "Rhode Island": "Northeast", "Connecticut": "Northeast",
    "New York": "Northeast", "New Jersey": "Northeast", "Pennsylvania": "Northeast",

    # Midwest
    "Ohio": "Midwest", "Indiana": "Midwest", "Illinois": "Midwest",
    "Michigan": "Midwest", "Wisconsin": "Midwest", "Minnesota": "Midwest",
    "Iowa": "Midwest", "Missouri": "Midwest", "North Dakota": "Midwest",
    "South Dakota": "Midwest", "Nebraska": "Midwest", "Kansas": "Midwest",

    # South
    "Delaware": "South", "Maryland": "South", "Virginia": "South", "West Virginia": "South",
    "Kentucky": "South", "North Carolina": "South", "South Carolina": "South",
    "Georgia": "South", "Florida": "South", "Alabama": "South", "Mississippi": "South",
    "Tennessee": "South", "Arkansas": "South", "Louisiana": "South", "Texas": "South",
    "Oklahoma": "South",

    # West
    "Montana": "West", "Idaho": "West", "Wyoming": "West", "Colorado": "West",
    "New Mexico": "West", "Arizona": "West", "Utah": "West", "Nevada": "West",
    "California": "West", "Oregon": "West", "Washington": "West", "Alaska": "West",
    "Hawaii": "West"
}

# Add Region column
df["Region"] = df["State"].map(region_mapping)

if df["Region"].isnull().any():
    print("⚠️ Some states are missing in region mapping!")

# --- Step 2: Aggregate by Region ---
df_region = df.groupby("Region", as_index=False)["Number of Manufacturers"].sum()

# Simulate Marketing Spend (random, proportional to manufacturers)
np.random.seed(42)
df_region["Marketing_Spend"] = df_region["Number of Manufacturers"] * np.random.uniform(0.8, 1.2, size=len(df_region))

# --- Step 3: Deep Learning Model ---
X = df_region[["Marketing_Spend"]].values  # input feature
y = df_region["Number of Manufacturers"].values  # target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
model = Sequential([
    Input(shape=(1,)),         # Input layer
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="linear")   # Output
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_scaled, y, epochs=50, verbose=0)

# Predictions
df_region["Predicted_Manufacturers"] = model.predict(X_scaled).flatten()

# --- Step 4: Visualization ---

# Bar Graph: Predicted Manufacturers by Region
plt.figure(figsize=(10,6))
plt.bar(df_region["Region"], df_region["Predicted_Manufacturers"], color="skyblue", alpha=0.8)
plt.title("Predicted Manufacturers by U.S. Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Predicted Manufacturers")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Area Graph: Compare Marketing Spend vs Predicted Manufacturers
plt.figure(figsize=(10,6))
plt.fill_between(df_region["Region"], df_region["Marketing_Spend"], color="orange", alpha=0.4, label="Marketing Spend")
plt.fill_between(df_region["Region"], df_region["Predicted_Manufacturers"], color="blue", alpha=0.4, label="Predicted Manufacturers")
plt.title("Region-wise Marketing Spend vs Predicted Manufacturers", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
