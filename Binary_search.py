import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import openpyxl

# -----------------------
# STEP 1: Load Data
# -----------------------
file_path = ".venv/Toy_Manufacturers_with_Sales_2005_2025.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("✅ Columns found:", df.columns.tolist())

# Adjust these based on your Excel file
year_col = "year"
sales_col = [c for c in df.columns if "sale" in c][0]   # auto-detect sales column

# -----------------------
# STEP 2: BST Implementation
# -----------------------
class Node:
    def __init__(self, year, sales):
        self.year = year
        self.sales = sales
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, year, sales):
        if self.root is None:
            self.root = Node(year, sales)
        else:
            self._insert(self.root, year, sales)

    def _insert(self, node, year, sales):
        if sales < node.sales:
            if node.left is None:
                node.left = Node(year, sales)
            else:
                self._insert(node.left, year, sales)
        else:
            if node.right is None:
                node.right = Node(year, sales)
            else:
                self._insert(node.right, year, sales)

    def inorder(self, node, result):
        if node:
            self.inorder(node.left, result)
            result.append((node.year, node.sales))
            self.inorder(node.right, result)

# Build BST
bst = BST()
for _, row in df.iterrows():
    bst.insert(row[year_col], row[sales_col])

# Get BST ranked sales
ranked_sales = []
bst.inorder(bst.root, ranked_sales)
ranked_df = pd.DataFrame(ranked_sales, columns=["Year", "Sales"])

# -----------------------
# STEP 3: Deep Learning (LSTM) Prediction
# -----------------------
scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(df[sales_col].values.reshape(-1,1))

X, y = [], []
for i in range(3, len(scaled_sales)):
    X.append(scaled_sales[i-3:i, 0])
    y.append(scaled_sales[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=100, batch_size=1, verbose=0)

# Predict future sales (next 10 years)
future_years = np.arange(df[year_col].max() + 1, df[year_col].max() + 11)
last_data = scaled_sales[-3:].reshape(1, 3, 1)
pred_sales = []
for _ in range(10):
    pred = model.predict(last_data, verbose=0)
    pred_sales.append(pred[0][0])
    last_data = np.append(last_data[:,1:,:], [[pred]], axis=1)

pred_sales = scaler.inverse_transform(np.array(pred_sales).reshape(-1,1)).flatten()

# -----------------------
# STEP 4: Visualization
# -----------------------
# Bar Graph - BST Ranked Sales
plt.figure(figsize=(10,5))
plt.bar(ranked_df["Year"].astype(str), ranked_df["Sales"], color="skyblue", edgecolor="black")
plt.xticks(rotation=45)
plt.title("Sales Ranking by Year (BST Order - Bar Graph)")
plt.ylabel("Sales (USD)")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("bst_bar_graph.png")
plt.show()

# Area Graph - Actual + Predicted
plt.figure(figsize=(10,5))
plt.fill_between(df[year_col], df[sales_col], color="orange", alpha=0.4, step="mid", label="Actual Sales")
plt.plot(df[year_col], df[sales_col], marker="o", color="red", linewidth=2)
plt.fill_between(future_years, pred_sales, color="green", alpha=0.3, step="mid", label="Predicted Sales")
plt.plot(future_years, pred_sales, marker="o", color="green", linewidth=2)
plt.title("Sales Trend (Actual + Predicted) - Slanted Area Graph")
plt.xlabel("Year")
plt.ylabel("Sales (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("sales_area_graph.png")
plt.show()

# -----------------------
# STEP 5: Save Results
# -----------------------
output_file = "sales_analysis.xlsx"
future_df = pd.DataFrame({"Year": future_years, "Predicted_Sales": pred_sales})

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    ranked_df.to_excel(writer, sheet_name="BST_Ranked_Sales", index=False)
    df.to_excel(writer, sheet_name="Actual_Sales", index=False)
    future_df.to_excel(writer, sheet_name="Predictions", index=False)

print(f"✅ Data saved to {output_file}")
print("✅ Graphs saved as bst_bar_graph.png and sales_area_graph.png")
