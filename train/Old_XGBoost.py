import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor  # <-- เปลี่ยนตรงนี้

# Load Data
df = pd.read_csv("../data/arl_ready_feature.csv")


# Prepare Features (X) & Target (y)
X = df.drop(
    columns=["passenger_origin", "is_festival", "temp_bin", "temp_range", "cloudcover"]
)

y = df["passenger_origin"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle Missing Values
X_train_cleaned = X_train.dropna()
y_train_cleaned = y_train[X_train_cleaned.index]

X_test_cleaned = X_test.dropna()
y_test_cleaned = y_test[X_test_cleaned.index]

# Normalize Features
scaler = StandardScaler()
scaler.fit(X_train_cleaned)

X_train_scaled = scaler.transform(X_train_cleaned)
X_test_scaled = scaler.transform(X_test_cleaned)

# Create & Train XGBoost Model
model = XGBRegressor(
    n_estimators=100,  # จำนวน tree
    learning_rate=0.1,  # step size
    max_depth=3,  # ความลึกของ tree
    random_state=42,
)
model.fit(X_train_scaled, y_train_cleaned)

# Make Predictions & Evaluate
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test_cleaned, y_pred)
mse = mean_squared_error(y_test_cleaned, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_cleaned, y_pred)

print("=== Model Evaluation ===")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# Save Model & Scaler
# joblib.dump(model, "../xgb_model.pkl")
# joblib.dump(scaler, "../scaler.pkl")
# print("Saved XGBoost model and scaler to files.")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_cleaned, y_pred, alpha=0.5, color="blue", label="Predicted")
plt.plot(
    [y_test_cleaned.min(), y_test_cleaned.max()],
    [y_test_cleaned.min(), y_test_cleaned.max()],
    color="red",
    lw=2,
    label="y = x (Actual = Predicted)",
)
plt.xlabel("Actual Passenger Count")
plt.ylabel("Predicted Passenger Count")
plt.title("Actual vs Predicted - XGBoost")
plt.legend()
plt.grid(True)
plt.show()
