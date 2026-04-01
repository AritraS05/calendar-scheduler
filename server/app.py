import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="Meta Calendar Scheduler")

calendar_events = []

class ResetResponse(BaseModel):
    status: str
    message: str

@app.get("/")
async def root():
    return {"message": "Meta Calendar Scheduler is running"}

@app.post("/reset", response_model=ResetResponse)
async def reset():
    global calendar_events
    calendar_events = []
    return ResetResponse(status="success", message="Environment reset successfully")


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()