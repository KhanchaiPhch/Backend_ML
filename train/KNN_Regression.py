import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("../data/arl_ready_feature.csv")
# df.head()   # ใช้ดูตัวอย่างข้อมูล (optional)

# Prepare Features (X) & Target (y)
X = df.drop(
    columns=["passenger_origin", "is_festival", "temp_bin", "temp_range", "cloudcover"]
)
y = df["passenger_origin"]

# Train/Test Split
# แบ่ง 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle Missing Values
X_train_cleaned = X_train.dropna()
y_train_cleaned = y_train[X_train_cleaned.index]

X_test_cleaned = X_test.dropna()
y_test_cleaned = y_test[X_test_cleaned.index]

# Normalize Features
# ใช้ StandardScaler เพื่อให้ mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_cleaned)

X_train_scaled = scaler.transform(X_train_cleaned)  # Transform train
X_test_scaled = scaler.transform(X_test_cleaned)  # Transform test

# Create & Train KNN Model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train_cleaned)  # Train ด้วยข้อมูล normalized

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
# joblib.dump(model, "../knn_model.pkl")  # บันทึกโมเดล
# joblib.dump(scaler, "../scaler.pkl")  # บันทึก scaler เพื่อใช้ normalize ตอน deploy
# print("Saved KNN model and scaler to files.")


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
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
