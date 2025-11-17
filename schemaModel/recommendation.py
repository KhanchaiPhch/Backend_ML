from pydantic import BaseModel

class Recommendation(BaseModel):
    station:str
    date: str          # "YYYY-MM-DD"
