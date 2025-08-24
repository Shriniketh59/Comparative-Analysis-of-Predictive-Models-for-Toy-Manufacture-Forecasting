import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# ------------------------
# Load Data
# ------------------------
file_path = ".venv/Toy_Manufacturers_with_Sales_2005_2025.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Use correct columns
df = df[['Year', 'Marketing Sales Value (USD)']].dropna()
df['Year'] = df['Year'].astype(int)

# ------------------------
# Deep Learning Model (Predict Sales)
# ------------------------
X = df[['Year']].values
y = df['Marketing Sales Value (USD)'].values

# Normalize
X_min, X_max = X.min(), X.max()
X_scaled = (X - X_min) / (X_max - X_min)

# Define simple neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(16, activation='relu'),
    Dense(1)  # output layer
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=200, verbose=0)

# Predictions
y_pred = model.predict(X_scaled).flatten()

# ------------------------
# Merge Sort Implementation
# ------------------------
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] <= right[j][1]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Apply Merge Sort on Predictions
data = list(zip(df['Year'], y_pred))
sorted_merge = merge_sort(data)
sorted_df = pd.DataFrame(sorted_merge, columns=['Year', 'Predicted Sales'])

# Save Results
sorted_df.to_excel("Sorted_Predicted_Sales.xlsx", index=False)

# ------------------------
# Visualization
# ------------------------

# 1. Bar Chart
plt.figure(figsize=(10,6))
plt.bar(sorted_df['Year'].astype(str), sorted_df['Predicted Sales'])
plt.xticks(rotation=45)
plt.title("Merge Sort (Deep Learning Predictions) - Bar Chart")
plt.xlabel("Year (Sorted)")
plt.ylabel("Predicted Sales (USD)")
plt.tight_layout()
plt.show()

# 2. Area Chart
plt.figure(figsize=(10,6))
plt.fill_between(sorted_df['Year'], sorted_df['Predicted Sales'], alpha=0.4)
plt.plot(sorted_df['Year'], sorted_df['Predicted Sales'], marker='o')
plt.title("Merge Sort (Deep Learning Predictions) - Area Chart")
plt.xlabel("Year (Sorted)")
plt.ylabel("Predicted Sales (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

