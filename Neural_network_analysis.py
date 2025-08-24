import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ------------------------
# 1. Load Data
# ------------------------
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)

# ------------------------
# 2. Prepare Data
# ------------------------
df = df.sort_values(by=["State", "Year"])

sequence_length = 5  # use past 5 years to predict next year
X, y, states, years = [], [], [], []

for state in df["State"].unique():
    data = df[df["State"] == state]["Number of Manufacturers"].values

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # past 5 years
        y.append(data[i + sequence_length])  # next year
        states.append(state)
        years.append(df[df["State"] == state]["Year"].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Normalize for NN
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ------------------------
# 3. Build Neural Network
# ------------------------
model = Sequential([
    Dense(128, activation="relu", input_shape=(sequence_length,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)  # regression output
])
model.compile(optimizer="adam", loss="mse")

# ------------------------
# 4. Train
# ------------------------
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# ------------------------
# 5. Predict
# ------------------------
y_pred = model.predict(X).flatten()
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"✅ RMSE: {rmse:.2f}")

results_df = pd.DataFrame({
    "State": states,
    "Year": years,
    "Actual": y,
    "Predicted": y_pred
})
# Top 10 states by total manufacturers
top10_states = df.groupby("State")["Number of Manufacturers"].sum().nlargest(10)

# Plot bar chart
plt.figure(figsize=(12, 7))
bars = plt.bar(top10_states.index, top10_states.values, color=plt.cm.tab10.colors, edgecolor="black")

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval),
             ha='center', va='bottom', fontsize=12, fontweight="bold")

# Styling
plt.title("Top 10 States by Total Toy Manufacturers (2005–2016)", fontsize=18, fontweight="bold")
plt.xlabel("States", fontsize=14)
plt.ylabel("Number of Manufacturers", fontsize=14)
plt.xticks(rotation=30, fontsize=12, fontweight="bold")
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()
# Group data by Year and State
state_year = df.groupby(["Year", "State"])["Number of Manufacturers"].sum().reset_index()

# Pivot to wide format for area chart
pivot_data = state_year.pivot(index="Year", columns="State", values="Number of Manufacturers").fillna(0)

# Take Top 10 states overall
top10_states = df.groupby("State")["Number of Manufacturers"].sum().nlargest(10).index
pivot_top10 = pivot_data[top10_states]

# Plot area chart
plt.figure(figsize=(14, 8))
pivot_top10.plot.area(ax=plt.gca(), cmap="tab10", alpha=0.85)

# Labels & Styling
plt.title("Toy Manufacturers Trend by Top 10 States (2005–2016)", fontsize=18, fontweight="bold")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Manufacturers", fontsize=14)
plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)

plt.show()