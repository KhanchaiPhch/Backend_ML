from pydantic import BaseModel

class Recommendation(BaseModel):
    station:str
    date: str          # "YYYY-MM-DD"
    hour: int         # 0-23
