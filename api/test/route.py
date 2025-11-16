# main.py
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()

# โหลดโมเดลและ scaler
# model = joblib.load("xgb_model.pkl")  # หรือ knn_model.pkl
# scaler = joblib.load("scaler.pkl")

# def test(data: dict):
#     """
#     data: dict ที่รับมาจาก user เช่น {"feature1": value1, "feature2": value2, ...}
#     return: list ของค่าที่ทำนาย
#     """
#     # แปลง dict เป็น DataFrame
#     input_df = pd.DataFrame([data])

#     # Scale features
#     input_scaled = scaler.transform(input_df)

#     # ทำ Prediction
#     prediction = model.predict(input_scaled)
#     return prediction.tolist()

class PassengerFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@router.post("/test")
def predict(features: PassengerFeatures):
    try:
        data = features.dict()
        prediction = "result_here"

        return {"status": "OK", "prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
