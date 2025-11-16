from pydantic import BaseModel

class PassengerFeatures(BaseModel):
    station: str       # A1-A8
    date: str          # "YYYY-MM-DD"
    hour: int          # 5-23
