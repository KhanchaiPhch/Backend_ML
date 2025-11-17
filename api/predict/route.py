from fastapi import APIRouter, HTTPException
from schemaModel.passenger_features import PassengerFeatures
from schemaModel.passenger_forecast_3day import PassengerForecast3Day
from schemaModel.recommendation import Recommendation
import joblib
import numpy as np
import pandas as pd
from api.holiday.route import get_thai_holidays  # holiday api
from sqlalchemy.dialects.postgresql import insert
from db import SessionLocal
from models.predictions import Prediction
from config import MODEL_VERSION   # ถ้าเก็บ MODEL_VERSION ไว้ใน config.py
from datetime import datetime

model = joblib.load("new_XGBoost.pkl")
router = APIRouter()

# ค่าคงที่ใช้ร่วมกัน
STATION_COLS = ["is_A1", "is_A2", "is_A3", "is_A4", "is_A5", "is_A6", "is_A7", "is_A8"]
STATIONS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
WEEKDAY_MAP = [
    "is_Monday",
    "is_Tuesday",
    "is_Wednesday",
    "is_Thursday",
    "is_Friday",
    "is_Saturday",
    "is_Sunday",
]
HOUR_RANGE = range(5, 24)  # 05:00 - 23:00
PEAK_HOURS = [7, 8, 17, 18]

FEATURE_COLUMNS = [
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
    "is_holiday",
    "is_weekend",
    "rain_flag",
    "temp",
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

# ------------------ 1) ทำนายปกติทีละจุด ------------------ #
@router.post("/")
def predict_single(features: PassengerFeatures):
    try:
        # --- Date encode ---
        date = pd.to_datetime(features.date)
        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")

        # ดึงวันหยุดเฉพาะวันเดียว
        holidays = get_thai_holidays(start_date=date_str, end_date=date_str)
        is_holiday = 1 if holidays else 0

        # --- Station encode ---
        station_data = {col: 0 for col in STATION_COLS}
        if features.station not in [col[3:] for col in STATION_COLS]:
            raise HTTPException(status_code=400, detail="Invalid station")
        station_data[f"is_{features.station}"] = 1

        # --- Weekday encode ---
        weekday_data = {col: 0 for col in WEEKDAY_MAP}
        weekday_data[WEEKDAY_MAP[date.weekday()]] = 1

        is_weekend = 1 if date.weekday() >= 5 else 0

        # --- Hour encode + peak time ---
        hour_cols = [f"is_hour_{h}" for h in HOUR_RANGE]
        hour_data = {col: 0 for col in hour_cols}
        if features.hour not in HOUR_RANGE:
            raise HTTPException(status_code=400, detail="Invalid hour")
        hour_data[f"is_hour_{features.hour}"] = 1

        is_peak_time = 1 if features.hour in PEAK_HOURS else 0

        # --- รวม dict ---
        data_dict = {}
        data_dict.update(weekday_data)
        data_dict.update(station_data)
        data_dict.update(
            {
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
                "rain_flag": 0,
                "temp": 30.0,
            }
        )
        data_dict.update(hour_data)
        data_dict["is_peak_time"] = is_peak_time

        # เติม 0 ถ้าคอลัมน์ไหนไม่มี
        for col in FEATURE_COLUMNS:
            if col not in data_dict:
                data_dict[col] = 0

        df_input = pd.DataFrame([data_dict])[FEATURE_COLUMNS]

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
                "model_version": MODEL_VERSION,
            },
            "prediction": float(prediction),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ 2) ทำนายล่วงหน้า 3 วัน (ทุกสถานี ทุกชั่วโมง) ------------------ #
@router.post("/3days")
def predict_3day(payload: PassengerForecast3Day):
    try:
        # แปลงวันที่เริ่มต้น
        base_date = pd.to_datetime(payload.date)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid date format, expected YYYY-MM-DD"
        )

    # เตรียมช่วงวันที่: day0, day1, day2
    dates = [base_date + pd.Timedelta(days=i) for i in range(3)]
    start_str = dates[0].strftime("%Y-%m-%d")
    end_str = dates[-1].strftime("%Y-%m-%d")

    # ดึงวันหยุดในช่วง 3 วัน
    holidays_raw = get_thai_holidays(start_date=start_str, end_date=end_str)
    # รองรับทั้งกรณีคืนเป็น dict {date: ...} หรือ list ["date1", "date2", ...]
    if isinstance(holidays_raw, dict):
        holiday_dates = set(holidays_raw.keys())
    else:
        holiday_dates = set(holidays_raw)

    rows = []
    meta = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        day_name = d.strftime("%A")

        # weekday one-hot
        weekday_data = {col: 0 for col in WEEKDAY_MAP}
        weekday_data[WEEKDAY_MAP[d.weekday()]] = 1

        is_weekend = 1 if d.weekday() >= 5 else 0
        is_holiday = 1 if date_str in holiday_dates else 0

        for station in STATIONS:
            # station one-hot
            station_data = {col: 0 for col in STATION_COLS}
            station_data[f"is_{station}"] = 1

            for hour in HOUR_RANGE:
                # hour one-hot
                hour_cols = [f"is_hour_{h}" for h in HOUR_RANGE]
                hour_data = {col: 0 for col in hour_cols}
                hour_data[f"is_hour_{hour}"] = 1

                is_peak_time = 1 if hour in PEAK_HOURS else 0

                data_dict = {}
                data_dict.update(weekday_data)
                data_dict.update(station_data)
                data_dict.update(
                    {
                        "is_holiday": is_holiday,
                        "is_weekend": is_weekend,
                        "rain_flag": 0,
                        "temp": 30.0,
                    }
                )
                data_dict.update(hour_data)
                data_dict["is_peak_time"] = is_peak_time

                # เติม 0 ให้ครบทุกคอลัมน์
                for col in FEATURE_COLUMNS:
                    if col not in data_dict:
                        data_dict[col] = 0

                rows.append([data_dict[col] for col in FEATURE_COLUMNS])
                meta.append(
                    {
                        "date": date_str,
                        "day_name": day_name,
                        "station": station,
                        "hour": hour,
                        "is_holiday": is_holiday,
                        "is_weekend": is_weekend,
                    }
                )

    # สร้าง DataFrame และทำนาย
    df_input = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    log_pred = model.predict(df_input.values)
    predictions = np.expm1(log_pred)

    results = []
    for m, pred in zip(meta, predictions):
        results.append({**m, "prediction": int(float(pred))})

    # ---------------- SAVE TO DB (UPSERT) ---------------- #
    db = SessionLocal()
    try:
        for r in results:
            stmt = insert(Prediction).values(
                station=r["station"],
                prediction_date=r["date"],        # string "YYYY-MM-DD" -> Date, SQLAlchemy แปลงให้ได้
                hour=r["hour"],
                prediction_passenger=r["prediction"],
                model_version=MODEL_VERSION,
            )

            # ถ้าซ้ำ (station + prediction_date + hour + model_version) → update prediction_passenger
            stmt = stmt.on_conflict_do_update(
                index_elements=["station", "prediction_date", "hour", "model_version"],
                set_={
                    "prediction_passenger": r["prediction"]
                    # ไม่แตะ created_at → เวลาแรกยังอยู่เหมือนเดิม
                },
            )

            db.execute(stmt)

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()
    # ----------------------------------------------------- #

    return {
        "status": "OK",
        "start_date": start_str,
        "end_date": end_str,
        "total_points": len(results),
        "results": results,
        "model_version": MODEL_VERSION,
    }

@router.post("/predictions")
def get_prediction_from_db(features: PassengerFeatures):
    """
    ดึงผล prediction จาก DB แบบ 5 ชั่วโมงถัดจาก hour ที่ส่งมา
    เช่น hour=6 → ดึง 6,7,8,9,10
    """
    # แปลงเป็น date object
    try:
        prediction_date = datetime.strptime(features.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format, expected YYYY-MM-DD",
        )

    start_hour = features.hour
    end_hour = min(start_hour + 4, 23)   # จำกัดไม่เกิน 23

    db = SessionLocal()
    try:
        records = (
            db.query(Prediction)
            .filter(
                Prediction.station == features.station,
                Prediction.prediction_date == prediction_date,
                Prediction.hour >= start_hour,
                Prediction.hour <= end_hour,
            )
            .order_by(Prediction.hour.asc())
            .all()
        )

        if not records:
            raise HTTPException(
                status_code=404,
                detail="No prediction results found for given station/date/hour range",
            )

        # format response
        response = []
        for r in records:
            response.append({
                "station": r.station,
                "prediction_date": r.prediction_date.strftime("%Y-%m-%d"),
                "hour": r.hour,
                "prediction_passenger": r.prediction_passenger,
                "model_version": r.model_version,
            })

        return {
            "requested_hour": features.hour,
            "returned_hours": [r["hour"] for r in response],
            "results": response,
        }

    finally:
        db.close()

@router.post("/recommendation")
def recommend_low_density(features: Recommendation):
    """
    แนะนำ 3 ชั่วโมงที่มีจำนวนผู้โดยสารน้อยที่สุด
    สำหรับสถานี + วันที่ ที่ส่งมา
    """
    # แปลง string -> date object
    try:
        prediction_date = datetime.strptime(features.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format, expected YYYY-MM-DD",
        )

    db = SessionLocal()
    try:
        # ดึงทุกชั่วโมงของสถานี+วันที่นั้น แล้ว sort ตามจำนวนผู้โดยสารจากน้อยไปมาก
        records = (
            db.query(Prediction)
            .filter(
                Prediction.station == features.station,
                Prediction.prediction_date == prediction_date,
            )
            .order_by(Prediction.prediction_passenger.asc())
            .limit(3)  # เอาแค่ 3 อันดับแรกที่คนน้อยที่สุด
            .all()
        )

        if not records:
            raise HTTPException(
                status_code=404,
                detail="No prediction results found for given station/date",
            )

        # format response
        results = []
        for r in records:
            results.append({
                "station": r.station,
                "prediction_date": r.prediction_date.strftime("%Y-%m-%d"),
                "hour": r.hour,
                "prediction_passenger": r.prediction_passenger,
                "model_version": r.model_version,
            })

        return {
            "station": features.station,
            "date": features.date,
            "count": len(results),
            "results": results,
        }

    finally:
        db.close()