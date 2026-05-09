from sqlalchemy import Column, Integer, String, Float
from database import Base

class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(String, index=True)
    message = Column(String)
    type = Column(String)

class TelemetryHistory(Base):
    __tablename__ = "telemetry_history"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(String, index=True)
    mae = Column(Float)
    threshold = Column(Float)
    status = Column(String) # 'Green', 'Yellow', 'Red'
