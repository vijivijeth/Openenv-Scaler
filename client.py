from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import SREAction, SREObservation, SREState


class SREResponseGymClient(EnvClient[SREAction, SREObservation, SREState]):
    """Typed OpenEnv client for the SRE-ResponseGym environment."""

    def _step_payload(self, action: SREAction | Dict[str, Any] | str) -> Dict[str, Any]:
        if isinstance(action, str):
            return {"action": action}
        if isinstance(action, dict):
            return action
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SREObservation]:
        observation_data = payload.get("observation", payload)
        observation = SREObservation(**observation_data)
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SREState:
        return SREState(**payload)
