from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import datetime

app = FastAPI()

class CalendarAction(BaseModel):
    action_type: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    guest_email: Optional[str] = None

class CalendarObservation(BaseModel):
    current_schedule: List[Dict]
    last_action_status: str
    last_action_error: bool

class CalendarReward(BaseModel):
    score: float
    is_partial: bool

class StepResponse(BaseModel):
    observation: CalendarObservation
    reward: float
    done: bool
    info: dict

class CalendarEnv:
    def __init__(self):
        self.bookings = []
        self.target_booked = False
        self.steps_taken = 0

    def reset(self):
        self.bookings = []
        self.target_booked = False
        self.steps_taken = 0
        return self.get_observation("Environment reset.", False)

    def get_observation(self, status: str, error: bool):
        return CalendarObservation(
            current_schedule=self.bookings,
            last_action_status=status,
            last_action_error=error
        )

    def step(self, action: CalendarAction):
        self.steps_taken += 1
        reward = 0.0
        done = False
        
        if action.action_type == "view_schedule":
            reward = 0.1 
            status_msg = f"Schedule viewed. Current bookings: {self.bookings}"
            obs = self.get_observation(status_msg, False)
            
        elif action.action_type == "book_slot":
            conflict = any(b["start"] == action.start_time for b in self.bookings)
            
            if conflict:
                reward = 0.0
                obs = self.get_observation("Conflict! Time slot already booked.", True)
            else:
                self.bookings.append({"start": action.start_time, "end": action.end_time, "email": action.guest_email})
                reward = 1.0 
                done = True
                obs = self.get_observation(f"Successfully booked {action.start_time}", False)
                
        else:
            obs = self.get_observation("Unknown action. Use 'view_schedule' or 'book_slot'.", True)
            reward = 0.0
            
        if self.steps_taken >= 10:
            done = True
            
        return StepResponse(observation=obs, reward=reward, done=done, info={"steps": self.steps_taken})

env_instance = CalendarEnv()

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/reset", response_model=CalendarObservation)
def reset_env():
    return env_instance.reset()

@app.post("/step", response_model=StepResponse)
def step_env(action: CalendarAction):
    return env_instance.step(action)

@app.get("/state", response_model=CalendarObservation)
def get_state():
    return env_instance.get_observation("Current state queried", False)