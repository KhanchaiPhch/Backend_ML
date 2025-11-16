from fastapi import APIRouter, HTTPException
from schemaModel.passenger_features import PassengerFeatures
import joblib
import numpy as np
import pandas as pd

# โหลดโมเดล
model = joblib.load("new_XGBoost.pkl")

router = APIRouter()


@router.post("/predict")
def predict(features: PassengerFeatures):
    try:
        # --- 1. Station encode ---
        station_cols = [
            "is_A1",
            "is_A2",
            "is_A3",
            "is_A4",
            "is_A5",
            "is_A6",
            "is_A7",
            "is_A8",
        ]
        station_data = {col: 0 for col in station_cols}
        if features.station in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]:
            station_data[f"is_{features.station}"] = 1
        else:
            raise HTTPException(status_code=400, detail="Invalid station")

        # --- 2. Date encode ---
        date = pd.to_datetime(features.date)
        weekday_map = [
            "is_Monday",
            "is_Tuesday",
            "is_Wednesday",
            "is_Thursday",
            "is_Friday",
            "is_Saturday",
            "is_Sunday",
        ]
        weekday_data = {col: 0 for col in weekday_map}
        weekday_data[weekday_map[date.weekday()]] = 1

        is_weekend = 1 if date.weekday() >= 5 else 0
        is_holiday = 0  # สามารถเพิ่ม lookup วันหยุดจริงได้
        is_festival = 0  # สามารถเพิ่ม lookup วัน festival จริงได้

        # --- 3. Hour encode ---
        hour_cols = [f"is_hour_{h}" for h in range(5, 24)]
        hour_data = {col: 0 for col in hour_cols}
        if features.hour in range(5, 24):
            hour_data[f"is_hour_{features.hour}"] = 1
        else:
            raise HTTPException(status_code=400, detail="Invalid hour")

        peak_hours = [7, 8, 17, 18]
        is_peak_time = 1 if features.hour in peak_hours else 0

        # --- 4. รวม dictionary ---
        data_dict = {}
        data_dict.update(station_data)
        data_dict.update(weekday_data)
        data_dict.update(
            {
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
                "is_festival": is_festival,
                "rain_flag": 0,  # default
                "cloudcover": 50,  # default
                "temp": 30.0,  # default
            }
        )
        data_dict.update(hour_data)
        data_dict["is_peak_time"] = is_peak_time

        # --- 5. เรียง column ให้ตรงกับตอนเทรน ---
        feature_columns = [
            "is_Monday",
            "is_Tuesday",
            "is_Wednesday",
            "is_Thursday",
            "is_Friday",
            "is_Saturday",
            "is_Sunday",
            "is_A1",
            "is_A2",
            "is_A3",
            "is_A4",
            "is_A5",
            "is_A6",
            "is_A7",
            "is_A8",
            "is_weekend",
            "cloudcover",
            "temp",
            "temp_range",
            "is_hour_5",
            "is_hour_6",
            "is_hour_7",
            "is_hour_8",
            "is_hour_9",
            "is_hour_10",
            "is_hour_11",
            "is_hour_12",
            "is_hour_13",
            "is_hour_14",
            "is_hour_15",
            "is_hour_16",
            "is_hour_17",
            "is_hour_18",
            "is_hour_19",
            "is_hour_20",
            "is_hour_21",
            "is_hour_22",
            "is_hour_23",
            "is_peak_time",
        ]

        # เพิ่ม column ที่ขาด
        for col in feature_columns:
            if col not in data_dict:
                data_dict[col] = 0

        df_input = pd.DataFrame([data_dict])[feature_columns]

        # --- 6. ทำนาย log1p passenger ---
        log_pred = model.predict(df_input.values)
        prediction = np.expm1(log_pred[0])

        return {"status": "OK", "prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
