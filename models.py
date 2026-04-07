from pydantic import BaseModel
from typing import Optional


class Action(BaseModel):
    action: str


class Observation(BaseModel):
    task_id: str
    description: str
    services: dict
    alerts: list
    logs: list
    step: int
    max_steps: int


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class State(BaseModel):
    episode_id: str
    step_count: int
    done: bool
    services: dict
    alerts: list
    logs: list


class GradeResult(BaseModel):
    final_score: float
    breakdown: dict
    actions_taken: list
    steps_used: int
    max_steps: int