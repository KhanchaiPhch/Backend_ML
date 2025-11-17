from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import Base, engine
from models.predictions import Prediction
from api.predict.route import router as predict_router

# สร้าง table ถ้ายังไม่มี
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ML Server API")

# CORS
origins = [
    "http://localhost:8080",
    "https://jirawatchn.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # frontend ที่อนุญาต
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/predict")
