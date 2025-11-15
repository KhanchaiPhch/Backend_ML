#  โหลด data, ทำ cleaning, split, เทรนโมเดล KNN, save .pkl
# 🔹 Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('../data/arl_ready_feature.csv')
# df.head()

# 🔹 Step 2: เตรียมข้อมูล
# ตัวอย่าง: X คือ features ทั้งหมด ยกเว้น passenger_origin
# ลบคอลัมน์ที่ไม่ใช่ตัวเลขออก
X = df.drop(columns=['passenger_origin', 'is_festival','temp_bin', 'temp_range', 'cloudcover'])

y = df['passenger_origin']

# แบ่งข้อมูลเป็น 80% สำหรับเทรน, 20% สำหรับเทสต์
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 3: สร้างและเทรนโมเดล KNN Regression

# n_neighbors = จำนวนเพื่อนบ้าน (k)
model = KNeighborsRegressor(n_neighbors=5)

# ลบแถวที่มีค่า NaN ออกจากข้อมูลสำหรับเทรนและเทสต์
X_train_cleaned = X_train.dropna()
y_train_cleaned = y_train[X_train_cleaned.index]

X_test_cleaned = X_test.dropna()
y_test_cleaned = y_test[X_test_cleaned.index]


# เทรนโมเดลด้วยข้อมูลที่ไม่มีค่า NaN
model.fit(X_train_cleaned, y_train_cleaned)


# บันทึกโมเดล KNN ลงไฟล์ .pkl
# joblib.dump(model, '../model/knn_model.pkl')
# joblib.dump(model, 'knn_model.pkl')
joblib.dump(model, '../knn_model.pkl')
print("Saved KNN model to knn_model.pkl")


# 🔹 Step 4: ทำนายผล (Prediction)
# y_pred = model.predict(X_test)


# 🔹 Step 5: ประเมินผล (Evaluation)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"MAE : {mae:.2f}")
# print(f"MSE : {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R²  : {r2:.2f}")



# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted')
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          color='red', lw=2, label='Linear y = x (Actual = Predicted)')
# plt.xlabel("(Actual)")
# plt.ylabel("(Predicted)")
# plt.title("Actual vs Predicted")
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(8,6))
# sns.kdeplot(y_test, label='Actual', fill=True)
# sns.kdeplot(y_pred, label='Predicted', fill=True)
# plt.title("Distribution of Actual vs Predicted")
# plt.legend()
# plt.show()


# ค่าที่น้องได้
metrics = {
    'MAE': 82.71,
    'MSE': 15640.74,
    'RMSE': 125.06,
    'R²': 0.83
}


# colors = []
# for key, value in metrics.items():
#     if key == 'R²':
#         if value >= 0.8:
#             colors.append('green')
#         elif value >= 0.5:
#             colors.append('orange')
#         else:
#             colors.append('red')
#     else:
#         if value <= ranges[key][1] * 0.5:
#             colors.append('green')
#         elif value <= ranges[key][1] * 0.8:
#             colors.append('orange')
#         else:
#             colors.append('red')

# plt.figure(figsize=(8,5))
# plt.barh(list(metrics.keys()), list(metrics.values()), color=colors)
# plt.title('Regression Model Evaluation Overview')
# plt.xlabel('Score / Error Value')

# for i, (metric, value) in enumerate(metrics.items()):
#     plt.text(value, i, f'  {value:.2f}', va='center', fontsize=10)

# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.show()
