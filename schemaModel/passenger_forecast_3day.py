from pydantic import BaseModel

class PassengerForecast3Day(BaseModel):
    date: str          # "YYYY-MM-DD"
