import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = ".venv/Toy_Manufacturers_with_Sales_2005_2025.xlsx"
df = pd.read_excel(file_path)

# Aggregate yearly total sales across all states
yearly_sales = df.groupby("Year")["Marketing Sales Value (USD)"].sum().reset_index()
years = yearly_sales["Year"].values
sales = yearly_sales["Marketing Sales Value (USD)"].values.reshape(-1, 1)

# -----------------------------
# 2. Deep Learning (LSTM) Forecast
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(sales)

def create_sequences(dataset, look_back=3):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_sequences(scaled_sales, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

# Forecast 2026–2030
future_years = np.arange(2026, 2031)
dl_predictions = []
last_seq = scaled_sales[-look_back:].reshape(1, look_back, 1)

for _ in range(len(future_years)):
    pred = model.predict(last_seq, verbose=0)
    dl_predictions.append(pred[0, 0])
    pred_reshaped = pred.reshape(1, 1, 1)
    last_seq = np.append(last_seq[:, 1:, :], pred_reshaped, axis=1)

dl_predictions = scaler.inverse_transform(np.array(dl_predictions).reshape(-1, 1))

# -----------------------------
# 3. Integral (Exponential) Forecast
# -----------------------------
log_sales = np.log(sales.flatten())
coeffs = np.polyfit(years, log_sales, 1)
b, a = coeffs

def sales_forecast_integral(year):
    return np.exp(a + b * year)

integral_predictions = sales_forecast_integral(future_years)

# -----------------------------
# 4. Clean Results
# -----------------------------
dl_predictions = np.nan_to_num(dl_predictions.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
integral_predictions = np.nan_to_num(integral_predictions.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------
# 5. Combine Results
# -----------------------------
forecast_df = pd.DataFrame({
    "Year": future_years,
    "Deep Learning Forecast (USD)": np.round(dl_predictions, 2),
    "Integral Forecast (USD)": np.round(integral_predictions, 2)
})

output_file = ".venv/Sales_Forecast_2026_2030_Comparison.xlsx"
forecast_df.to_excel(output_file, index=False)

print("✅ Forecast saved to", output_file)
print(forecast_df)

# -----------------------------
# 6. Pie Chart Visualization
# -----------------------------
plt.figure(figsize=(8, 8))

# Sum of forecasts for each method
dl_total = dl_predictions.sum()
integral_total = integral_predictions.sum()

labels = ["Sales Forecast Probability (2026-2030)", " Current Sales Forecast (2026-2030)"]
sizes = [dl_total, integral_total]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["skyblue", "lightgreen"])
plt.title("Sales Forecast")
plt.show()
# -----------------------------
# 7. Area Graph (Yearly Forecasts)
# -----------------------------
plt.figure(figsize=(10, 6))
plt.stackplot(
    forecast_df["Year"],
    forecast_df["Deep Learning Forecast (USD)"],
    forecast_df["Integral Forecast (USD)"],
    labels=["Deep Learning Forecast", "Integral Forecast"],
    colors=["skyblue", "lightgreen"],
    alpha=0.8
)
plt.xlabel("Year")
plt.ylabel("Sales Forecast (USD)")
plt.title("Yearly Sales Forecast (2026–2030)")
plt.legend(loc="upper left")
plt.show()