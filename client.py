import requests
from models import Action, StepResult, Observation, State


class SREResponseGymClient:

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "task_easy") -> StepResult:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        response.raise_for_status()
        data = response.json()
        return StepResult(
            observation=Observation(**data["observation"]),
            reward=0.0,
            done=False,
            info={"message": data.get("message", "")}
        )

    def step(self, action: str) -> StepResult:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action}
        )
        response.raise_for_status()
        data = response.json()
        return StepResult(
            observation=Observation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"]
        )

    def state(self) -> State:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()
        return State(**data)

    def grade(self) -> dict:
        response = requests.get(f"{self.base_url}/grade")
        response.raise_for_status()
        return response.json()