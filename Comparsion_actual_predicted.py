import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_excel("Week 39 - US Toy Manufacturers - 2005 to 2016.xlsx")

# Rename columns for consistency
df.rename(columns={"Number of Manufacturers": "Manufacturers"}, inplace=True)

# ==========================
# 2. Feature Engineering
# ==========================
df["Prev_Manufacturers"] = df.groupby("State")["Manufacturers"].shift(1)
df["Demand_Increase"] = (df["Manufacturers"] > df["Prev_Manufacturers"]).astype(int)
df = df.dropna()

X = df[["Year", "Prev_Manufacturers"]]
y = df["Demand_Increase"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ==========================
# 3. Train Models
# ==========================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]  # Prob of increase

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# ==========================
# 4. Aggregate Probabilities
# ==========================
rf_increase = rf_probs.mean()
rf_decrease = 1 - rf_increase

xgb_increase = xgb_probs.mean()
xgb_decrease = 1 - xgb_increase

# ==========================
# 5. Pie Charts
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Random Forest Pie
axes[0].pie([rf_increase, rf_decrease],
            labels=["Increase", "Decrease"],
            autopct='%1.1f%%',
            colors=["skyblue", "lightcoral"],
            startangle=90)
axes[0].set_title("Random Forest Prediction")

# XGBoost Pie
axes[1].pie([xgb_increase, xgb_decrease],
            labels=["Increase", "Decrease"],
            autopct='%1.1f%%',
            colors=["orange", "lightgreen"],
            startangle=90)
axes[1].set_title("XGBoost Prediction")

plt.suptitle("Probability of Manufacturer Demand (Increase vs Decrease)", fontsize=14)
plt.show()
