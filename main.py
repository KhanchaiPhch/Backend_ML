# server.py
from fastapi import FastAPI
# from database import connect_db

from api.predict.route import router as predict_router
from api.test.route import router as test_api

app = FastAPI(title="ML Server API")

# connect database
# connect_db()

# include router
app.include_router(predict_router, prefix="/api/predict")
app.include_router(test_api, prefix="/api/test")
