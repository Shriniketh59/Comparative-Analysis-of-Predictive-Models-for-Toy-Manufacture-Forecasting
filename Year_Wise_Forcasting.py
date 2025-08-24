import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl as xl
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
file_path = ".venv/Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower()

year_col = [col for col in df.columns if "year" in col][0]
manu_col = [col for col in df.columns if "manufacturer" in col][0]

df = df[[year_col, manu_col]].dropna()
df.columns = ["Year", "Manufacturers"]
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Manufacturers"] = pd.to_numeric(df["Manufacturers"], errors="coerce")
df = df.dropna()

# Get top 10 years by number of manufacturers
top10_df = df.nlargest(10, "Manufacturers").sort_values("Year")

print("\nTop 10 Years by Manufacturers:")
print(top10_df)
X = df[["Year"]]
y = df["Manufacturers"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# =======================
# 3️⃣ PROFESSIONAL VISUALIZATIONS (TOP 10)
# =======================
sns.set_style("whitegrid")
plt.figure(figsize=(16, 12))

# Color palettes
bar_palette = sns.color_palette("tab10", n_colors=10)
line_color = "#2ca02c"  # green
area_color = "#1f77b4"  # blue

# 📊 Bar Chart
plt.subplot(2, 2, 1)
sns.barplot(x="Year", y="Manufacturers", data=top10_df, palette=bar_palette)
plt.title("Top 10 Years by Toy Manufacturers", fontsize=16, weight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Manufacturers")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 📈 Line Chart
plt.subplot(2, 2, 2)
sns.lineplot(x="Year", y="Manufacturers", data=top10_df, marker="o", color=line_color, linewidth=3)
plt.title("Trend of Top 10 Years", fontsize=16, weight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Manufacturers")
plt.grid(linestyle='--', alpha=0.7)

# 📉 Area Chart
plt.subplot(2, 2, 3)
plt.fill_between(top10_df["Year"], top10_df["Manufacturers"], color=area_color, alpha=0.4)
plt.plot(top10_df["Year"], top10_df["Manufacturers"], color=area_color, linewidth=2)
plt.title("Area Chart of Top 10 Years", fontsize=16, weight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Manufacturers")
plt.grid(linestyle='--', alpha=0.7)

# 🥧 Pie Chart
plt.subplot(2, 2, 4)
plt.pie(top10_df["Manufacturers"], labels=top10_df["Year"], autopct="%1.1f%%", startangle=140, colors=bar_palette)
plt.title("Manufacturers Distribution (Top 10 Years)", fontsize=16, weight='bold')

plt.tight_layout()
plt.show()
