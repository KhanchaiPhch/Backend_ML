from fastapi import APIRouter, HTTPException
from schemaModel.passenger_features import PassengerFeatures
import joblib
import numpy as np
import pandas as pd
from api.holiday.route import get_thai_holidays   # เรียกใช้งาน holiday_api.py

model = joblib.load("new_XGBoost.pkl")
router = APIRouter()

@router.post("/predict")
def predict(features: PassengerFeatures):
    try:
        holiday_dict = get_thai_holidays()   # ดึงวันหยุด

        # --- Station encode ---
        station_cols = ["is_A1","is_A2","is_A3","is_A4","is_A5","is_A6","is_A7","is_A8"]
        station_data = {col:0 for col in station_cols}
        if features.station not in [col[3:] for col in station_cols]:
            raise HTTPException(status_code=400, detail="Invalid station")
        station_data[f"is_{features.station}"] = 1

        # --- Date encode ---
        date = pd.to_datetime(features.date)
        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")

        weekday_map = ["is_Monday","is_Tuesday","is_Wednesday","is_Thursday",
                       "is_Friday","is_Saturday","is_Sunday"]
        weekday_data = {col:0 for col in weekday_map}
        weekday_data[weekday_map[date.weekday()]] = 1

        is_weekend = 1 if date.weekday() >= 5 else 0
        is_holiday = 1 if date_str in holiday_dict else 0
        holiday_name = holiday_dict.get(date_str, None)
        is_festival = 0

        # --- Hour encode + peak time ---
        hour_cols = [f"is_hour_{h}" for h in range(5,24)]
        hour_data = {col:0 for col in hour_cols}
        if features.hour not in range(5,24):
            raise HTTPException(status_code=400, detail="Invalid hour")
        hour_data[f"is_hour_{features.hour}"] = 1

        peak_hours = [7,8,17,18]
        is_peak_time = 1 if features.hour in peak_hours else 0

        # --- รวม dict ---
        data_dict = {}
        data_dict.update(weekday_data)
        data_dict.update(station_data)
        data_dict.update({
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "is_festival": is_festival,
            "rain_flag": 0,
            "cloudcover": 50.0,
            "temp": 30.0
        })
        data_dict.update(hour_data)
        data_dict["is_peak_time"] = is_peak_time

        # --- เรียง column ---
        feature_columns = [
            "is_Monday","is_Tuesday","is_Wednesday","is_Thursday","is_Friday",
            "is_Saturday","is_Sunday",
            "is_A1","is_A2","is_A3","is_A4","is_A5","is_A6","is_A7","is_A8",
            "is_holiday","is_weekend",
            "rain_flag","temp",
            "is_hour_5","is_hour_6","is_hour_7","is_hour_8","is_hour_9","is_hour_10",
            "is_hour_11","is_hour_12","is_hour_13","is_hour_14","is_hour_15",
            "is_hour_16","is_hour_17","is_hour_18","is_hour_19","is_hour_20",
            "is_hour_21","is_hour_22","is_hour_23",
            "is_peak_time"
        ]
        for col in feature_columns:
            if col not in data_dict:
                data_dict[col] = 0

        df_input = pd.DataFrame([data_dict])[feature_columns]

        # --- Predict ---
        log_pred = model.predict(df_input.values)
        prediction = np.expm1(log_pred[0])

        return {
            "status": "OK",
            "input": {
                "station": features.station,
                "date": features.date,
                "hour": features.hour,
                "day_name": day_name,
                "is_holiday": is_holiday,
                "holiday_name": holiday_name
            },
            "prediction": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
