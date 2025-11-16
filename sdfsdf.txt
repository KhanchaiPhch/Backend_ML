# main.py
import joblib
import pandas as pd

# โหลดโมเดลและ scaler
model = joblib.load('xgb_model.pkl')     # หรือ knn_model.pkl
scaler = joblib.load('scaler.pkl')

def predict_passenger(data: dict):
    """
    data: dict ที่รับมาจาก user เช่น {"feature1": value1, "feature2": value2, ...}
    return: list ของค่าที่ทำนาย
    """
    # แปลง dict เป็น DataFrame
    input_df = pd.DataFrame([data])

    # Scale features
    input_scaled = scaler.transform(input_df)

    # ทำ Prediction
    prediction = model.predict(input_scaled)

    return prediction.tolist()
