from sqlalchemy import Column, Integer, String, Date, TIMESTAMP, text, UniqueConstraint
from db import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    station = Column(String(5), nullable=False)
    prediction_date = Column(Date, nullable=False)
    hour = Column(Integer, nullable=False)
    prediction_passenger = Column(Integer, nullable=False)
    model_version = Column(String(50), nullable=False)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint("station", "prediction_date", "hour", "model_version"),
    )
