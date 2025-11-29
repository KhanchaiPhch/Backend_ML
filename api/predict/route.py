from fastapi import APIRouter, HTTPException
from schemaModel.passenger_features import PassengerFeatures
from schemaModel.passenger_forecast_3day import PassengerForecast3Day
from schemaModel.recommendation import Recommendation
from schemaModel.overview import PassengerOverview

import joblib
import numpy as np
import pandas as pd
from api.holiday.route import get_thai_holidays  # holiday api
from sqlalchemy.dialects.postgresql import insert
from db import SessionLocal
from models.predictions import Prediction
from config import MODEL_VERSION  # ถ้าเก็บ MODEL_VERSION ไว้ใน config.py
from datetime import date, datetime

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


def build_model_input_for_hour(
    prediction_date: date, station: str, hour: int
) -> pd.DataFrame:
    """
    ใช้ logic เดียวกับ predict_single ในการ encode feature
    แต่รับ date เป็น date object + hour เป็น int
    """
    # --- Validate station & hour ---
    if station not in STATIONS:
        raise HTTPException(status_code=400, detail="Invalid station")

    if hour not in HOUR_RANGE:
        raise HTTPException(status_code=400, detail="Invalid hour")

    # --- Date & holiday encode ---
    date_str = prediction_date.strftime("%Y-%m-%d")
    holidays = get_thai_holidays(start_date=date_str, end_date=date_str)
    is_holiday = 1 if holidays else 0

    # --- Weekday encode ---
    weekday_data = {col: 0 for col in WEEKDAY_MAP}
    weekday_data[WEEKDAY_MAP[prediction_date.weekday()]] = 1
    is_weekend = 1 if prediction_date.weekday() >= 5 else 0

    # --- Station encode ---
    station_data = {col: 0 for col in STATION_COLS}
    station_data[f"is_{station}"] = 1

    # --- Hour encode ---
    hour_cols = [f"is_hour_{h}" for h in HOUR_RANGE]
    hour_data = {col: 0 for col in hour_cols}
    hour_data[f"is_hour_{hour}"] = 1

    # --- Peak time flag ---
    is_peak_time = 1 if hour in PEAK_HOURS else 0

    # --- รวม dict เป็น 1 แถว ---
    data_dict = {}
    data_dict.update(weekday_data)
    data_dict.update(station_data)
    data_dict.update(
        {
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "rain_flag": 0,  # ตอนนี้ fix ไว้ก่อน
            "temp": 86.63,  # ตอนนี้ fix ไว้ก่อน
        }
    )
    data_dict.update(hour_data)
    data_dict["is_peak_time"] = is_peak_time

    # เผื่ออนาคต FEATURE_COLUMNS มีเพิ่ม/ลด
    for col in FEATURE_COLUMNS:
        data_dict.setdefault(col, 0)

    return pd.DataFrame([data_dict])[FEATURE_COLUMNS]


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
                "temp": 86.63,
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


@router.post("/predictions")
def get_prediction_from_db(features: PassengerFeatures):
    """
    ดึงผล prediction แบบต่อเนื่อง 5 ชั่วโมงจาก hour ที่ส่งมา
    เช่น hour=6 → ดึง 6,7,8,9,10
    ถ้ามีใน DB แล้วใช้ค่าจาก DB
    ถ้าไม่มีใน DB ให้เรียก model ทำนายสดแทน
    """
    # 1) แปลง date string -> date object
    try:
        prediction_date = datetime.strptime(features.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format, expected YYYY-MM-DD",
        )

    # 2) validate station & hour เบื้องต้น
    if features.station not in STATIONS:
        raise HTTPException(status_code=400, detail="Invalid station")

    if features.hour not in HOUR_RANGE:
        raise HTTPException(status_code=400, detail="Invalid hour")

    # 3) คำนวณช่วงชั่วโมงที่จะดึง/ทำนาย
    start_hour = features.hour
    end_hour = min(start_hour + 4, 23)  # จำกัดไม่เกิน 23
    target_hours = list(range(start_hour, end_hour + 1))

    db = SessionLocal()
    try:
        # 4) ดึงข้อมูลที่มีอยู่แล้วใน DB สำหรับช่วงชั่วโมงนี้
        existing_records = (
            db.query(Prediction)
            .filter(
                Prediction.station == features.station,
                Prediction.prediction_date == prediction_date,
                Prediction.hour.in_(target_hours),
            )
            .order_by(Prediction.hour.asc())
            .all()
        )

        # map hour -> record เพื่อเช็คว่าชั่วโมงไหนมีอยู่แล้ว
        records_by_hour = {r.hour: r for r in existing_records}

        results = []

        for h in target_hours:
            if h in records_by_hour:
                # 5.1 ใช้ค่าที่มีอยู่ใน DB
                r = records_by_hour[h]
                results.append(
                    {
                        "station": r.station,
                        "prediction_date": r.prediction_date.strftime("%Y-%m-%d"),
                        "hour": r.hour,
                        "prediction_passenger": r.prediction_passenger,
                        "model_version": r.model_version,
                        "source": "db",  # optional: บอกว่าเอามาจาก DB
                    }
                )
            else:
                # 5.2 ไม่มีใน DB → เรียก model ทำนายสด
                df_input = build_model_input_for_hour(
                    prediction_date=prediction_date,
                    station=features.station,
                    hour=h,
                )

                log_pred = model.predict(df_input.values)
                prediction_value = float(np.expm1(log_pred[0]))

                results.append(
                    {
                        "station": features.station,
                        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
                        "hour": h,
                        "prediction_passenger": prediction_value,
                        "model_version": MODEL_VERSION,
                        "source": "model",  # optional: บอกว่าเป็น live prediction
                    }
                )

                new_record = Prediction(
                    station=features.station,
                    prediction_date=prediction_date,
                    hour=h,
                    prediction_passenger=prediction_value,
                    model_version=MODEL_VERSION,
                )
                db.add(new_record)

        # commit เฉพาะในกรณีที่มีการเพิ่มข้อมูล
        db.commit()

        return {
            "requested_hour": features.hour,
            "returned_hours": [r["hour"] for r in results],
            "results": results,
        }

    finally:
        db.close()


@router.post("/recommendation")
def recommend_low_density(features: Recommendation):
    """
    แนะนำ 3 ชั่วโมงที่มีจำนวนผู้โดยสารน้อยที่สุด
    ภายในช่วงเวลาที่เลือก ±2 ชั่วโมง

    เงื่อนไข:
    - ใช้ช่วงชั่วโมง [center_hour-2, center_hour+2] ที่อยู่ใน HOUR_RANGE (5–23)
    - ถ้า DB มีบางชั่วโมงแล้ว → ใช้
    - ถ้าบางชั่วโมงยังไม่มี → ทำนายใหม่, เก็บลง DB ให้ครบช่วงก่อน
    - สุดท้ายเลือก 3 ชั่วโมงที่คนน้อยที่สุดจากช่วงนั้น
    """

    # 1) แปลง string -> date object
    try:
        prediction_date = datetime.strptime(features.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format, expected YYYY-MM-DD",
        )

    # 2) validate station
    if features.station not in STATIONS:
        raise HTTPException(status_code=400, detail="Invalid station")

    # 3) validate hour (อินพุต)
    if not (0 <= features.hour <= 23):
        raise HTTPException(
            status_code=400,
            detail="hour must be between 0 and 23",
        )

    center_hour = features.hour

    # 4) สร้างช่วง center_hour - 2 ถึง center_hour + 2 (ตาม requirement เดิม)
    raw_start_hour = max(center_hour - 2, 0)
    raw_end_hour = min(center_hour + 2, 23)

    # พิจารณาเฉพาะชั่วโมงที่โมเดลรองรับ (HOUR_RANGE = 5–23)
    candidate_hours = [
        h for h in range(raw_start_hour, raw_end_hour + 1)
        if h in HOUR_RANGE
    ]

    if len(candidate_hours) < 3:
        # ทางทฤษฎีไม่ควรเกิด เพราะที่ขอบ (5 หรือ 23) ก็ยังได้อย่างน้อย 3 ชม.
        raise HTTPException(
            status_code=400,
            detail="Not enough valid hours in range for recommendation.",
        )

    start_hour = min(candidate_hours)
    end_hour = max(candidate_hours)

    db = SessionLocal()
    try:
        # 5) ดึงข้อมูลที่มีอยู่แล้วใน DB สำหรับทุกชั่วโมงใน candidate_hours
        existing_records = (
            db.query(Prediction)
            .filter(
                Prediction.station == features.station,
                Prediction.prediction_date == prediction_date,
                Prediction.hour.in_(candidate_hours),
            )
            .all()
        )

        records_by_hour: dict[int, Prediction] = {r.hour: r for r in existing_records}

        # ชั่วโมงที่ยังไม่มีใน DB
        missing_hours = [h for h in candidate_hours if h not in records_by_hour]

        predicted_hours = set()

        # 6) ถ้ามี missing_hours → ทำนายให้ครบ และเก็บลง DB
        if missing_hours:
            for h in missing_hours:
                df_input = build_model_input_for_hour(
                    prediction_date=prediction_date,
                    station=features.station,
                    hour=h,
                )

                log_pred = model.predict(df_input.values)
                prediction_value = float(np.expm1(log_pred[0]))

                new_record = Prediction(
                    station=features.station,
                    prediction_date=prediction_date,
                    hour=h,
                    prediction_passenger=prediction_value,
                    model_version=MODEL_VERSION,
                )
                db.add(new_record)
                records_by_hour[h] = new_record
                predicted_hours.add(h)

            db.commit()

        # 7) ตอนนี้ต้องมีข้อมูลครบทุกชั่วโมงใน candidate_hours แล้ว
        all_records = [records_by_hour[h] for h in candidate_hours]

        # 8) เลือก 3 ชั่วโมงที่คนน้อยที่สุด
        all_records_sorted = sorted(
            all_records,
            key=lambda r: r.prediction_passenger,
        )
        top3 = all_records_sorted[:3]

        results = []
        for r in top3:
            source = "model" if r.hour in predicted_hours else "db"
            results.append(
                {
                    "station": r.station,
                    "prediction_date": r.prediction_date.strftime("%Y-%m-%d"),
                    "hour": r.hour,
                    "prediction_passenger": r.prediction_passenger,
                    "model_version": r.model_version,
                    "source": source,  # optional ไว้ debug ว่ามาจากไหน
                }
            )

        return {
            "station": features.station,
            "date": features.date,
            "center_hour": center_hour,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "count": len(results),  # ปกติควรเป็น 3 เสมอ
            "results": results,
        }

    finally:
        db.close()


@router.post("/overview")
def get_daily_overview(payload: PassengerOverview):
    """
    ดึงผล prediction ทั้งวันของสถานีและวันที่ที่เลือก
    - ใช้ทุกชั่วโมงที่โมเดลรองรับ: 5–23 (HOUR_RANGE)
    - ถ้ามีบางชั่วโมงใน DB แล้ว → ใช้จาก DB
    - ถ้าบางชั่วโมงยังไม่มี → ทำนายใหม่ให้ครบทุกชั่วโมงในวันนั้น แล้วเก็บลง DB ก่อนส่งออก
    """
    # 1) แปลง string → date object
    try:
        prediction_date = datetime.strptime(payload.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format, expected YYYY-MM-DD",
        )

    # 2) validate station
    if payload.station not in STATIONS:
        raise HTTPException(status_code=400, detail="Invalid station")

    # 3) ช่วงชั่วโมงที่ต้องการ overview ทั้งวัน = HOUR_RANGE (เช่น 5–23)
    candidate_hours = list(HOUR_RANGE)

    db = SessionLocal()
    try:
        # 4) ดึงข้อมูลที่มีอยู่แล้วใน DB สำหรับ station/date นี้
        existing_records = (
            db.query(Prediction)
            .filter(
                Prediction.station == payload.station,
                Prediction.prediction_date == prediction_date,
                Prediction.hour.in_(candidate_hours),
            )
            .all()
        )

        records_by_hour: dict[int, Prediction] = {r.hour: r for r in existing_records}

        # ชั่วโมงที่ยังไม่มีใน DB
        missing_hours = [h for h in candidate_hours if h not in records_by_hour]
        predicted_hours = set()

        # 5) ถ้ามี missing_hours → ทำนายให้ครบ และเก็บลง DB
        if missing_hours:
            for h in missing_hours:
                df_input = build_model_input_for_hour(
                    prediction_date=prediction_date,
                    station=payload.station,
                    hour=h,
                )

                log_pred = model.predict(df_input.values)
                prediction_value = float(np.expm1(log_pred[0]))

                new_record = Prediction(
                    station=payload.station,
                    prediction_date=prediction_date,
                    hour=h,
                    prediction_passenger=prediction_value,
                    model_version=MODEL_VERSION,
                )
                db.add(new_record)
                records_by_hour[h] = new_record
                predicted_hours.add(h)

            db.commit()

        # 6) ตอนนี้ต้องมีข้อมูลครบทุกชั่วโมงใน candidate_hours แล้ว
        all_records = [records_by_hour[h] for h in sorted(candidate_hours)]

        # format response
        results = []
        for r in all_records:
            source = "model" if r.hour in predicted_hours else "db"
            results.append(
                {
                    "station": r.station,
                    "prediction_date": r.prediction_date.strftime("%Y-%m-%d"),
                    "hour": r.hour,
                    "prediction_passenger": r.prediction_passenger,
                    "model_version": r.model_version,
                    "source": source,  # optional: ดูได้ว่ามาจาก DB หรือทำนายสด
                }
            )

        return {
            "station": payload.station,
            "prediction_date": payload.date,
            "hours": [r["hour"] for r in results],
            "count": len(results),
            "results": results,
        }

    finally:
        db.close()
