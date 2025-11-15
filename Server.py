# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import predict_passenger

# สร้าง FastAPI app
app = FastAPI(title="Passenger Prediction API")

# กำหนด schema ของ input
class PassengerFeatures(BaseModel):
    # ใส่ชื่อฟีเจอร์ที่โมเดลต้องการ
    feature1: float
    feature2: float
    feature3: float
    # ... เพิ่มตามจำนวนฟีเจอร์จริง

@app.post("/predict")
def predict(features: PassengerFeatures):
    try:
        data_dict = features.dict()  # แปลง Pydantic object เป็น dict
        prediction = predict_passenger(data_dict)
        return {"status": "OK", "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
