from pydantic import BaseModel

class PassengerOverview(BaseModel):
    station: str   # A1–A8
    date: str      # "YYYY-MM-DD"
