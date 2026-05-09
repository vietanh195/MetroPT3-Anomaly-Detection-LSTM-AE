from pydantic import BaseModel
from typing import Optional

class SystemLogBase(BaseModel):
    time: str
    message: str
    type: str

class SystemLogCreate(SystemLogBase):
    pass

class SystemLog(SystemLogBase):
    id: int
    class Config:
        from_attributes = True

class TelemetryHistoryBase(BaseModel):
    time: str
    mae: float
    threshold: float
    status: Optional[str] = None

class TelemetryHistoryCreate(TelemetryHistoryBase):
    pass

class TelemetryHistory(TelemetryHistoryBase):
    id: int
    class Config:
        from_attributes = True
