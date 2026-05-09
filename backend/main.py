from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from simulator import simulator
from datetime import datetime
from sqlalchemy.orm import Session
from database import engine, get_db, Base
import models
import schemas

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/scenario/{scenario_name}")
async def set_scenario(scenario_name: str):
    success = simulator.set_scenario(scenario_name)
    return {"status": "success" if success else "error", "scenario": scenario_name}

@app.get("/api/logs", response_model=list[schemas.SystemLog])
def read_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(models.SystemLog).order_by(models.SystemLog.id.desc()).offset(skip).limit(limit).all()
    return logs

@app.post("/api/logs", response_model=schemas.SystemLog)
def create_log(log: schemas.SystemLogCreate, db: Session = Depends(get_db)):
    db_log = models.SystemLog(**log.model_dump())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    # Giữ tối đa 1000 logs
    if db.query(models.SystemLog).count() > 1000:
        oldest = db.query(models.SystemLog).order_by(models.SystemLog.id.asc()).first()
        db.delete(oldest)
        db.commit()
    return db_log

@app.get("/api/history", response_model=list[schemas.TelemetryHistory])
def read_history(limit: int = 360, db: Session = Depends(get_db)):
    history = db.query(models.TelemetryHistory).order_by(models.TelemetryHistory.id.desc()).limit(limit).all()
    history.reverse() # Trả về mảng đúng thứ tự thời gian
    return history

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Get latest data and mocked AI result
            data = simulator.get_current_data()
            if data:
                # Save Telemetry History to DB
                db = next(get_db())
                current_time = datetime.now().strftime("%H:%M:%S")
                ai = data.get("ai_inference", {})
                mae = ai.get("mae_absolute", ai.get("mae", 0.0))
                threshold = ai.get("threshold", 0.035)
                status = ai.get("status", "Green")
                
                new_hist = models.TelemetryHistory(time=current_time, mae=mae, threshold=threshold, status=status)
                db.add(new_hist)
                db.commit()
                
                if db.query(models.TelemetryHistory).count() > 1000:
                    oldest = db.query(models.TelemetryHistory).order_by(models.TelemetryHistory.id.asc()).first()
                    db.delete(oldest)
                    db.commit()
                db.close()
                
                await websocket.send_json(data)
                
            sleep_time = 4.0 # Fixed update interval required by user
            await asyncio.sleep(sleep_time)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WebSocket Error:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
