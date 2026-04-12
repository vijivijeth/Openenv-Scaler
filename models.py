from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class SREAction(Action):
    """Text action for the incident simulator."""

    action: str = Field(
        ...,
        description=(
            "Action to execute. Examples: check_logs auth-api, "
            "rollback_deploy payment-service v3.2.0, "
            "post_status critical investigating checkout outage"
        ),
    )


class SREObservation(Observation):
    """Observation returned after each step."""

    task_id: str = Field(..., description="Current task identifier")
    description: str = Field(..., description="Task description and objective")
    services: dict = Field(
        ..., description="Map of service name to status: healthy/degraded/down/etc."
    )
    alerts: list = Field(..., description="List of active alerts with severity and message")
    logs: list = Field(..., description="Recent visible log lines and evidence")
    metrics: dict = Field(..., description="Current per-service metric snapshot")
    timeline: list = Field(..., description="Recent incident timeline events")
    customer_impact: dict = Field(..., description="Current customer impact summary")
    available_tools: list = Field(..., description="Supported action grammar examples")
    last_action_result: dict = Field(
        default_factory=dict,
        description="Structured result for the most recent action",
    )
    step: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    reward: float = Field(default=0.01, description="Reward from last action")
    done: bool = Field(default=False, description="Whether the episode has ended")


class SREState(State):
    """Full environment state, including internal grading context."""

    task_id: str = Field(default="", description="Active task ID")
    step_count: int = Field(default=0, description="Number of steps taken")
    done: bool = Field(default=False, description="Whether the episode is complete")
    scenario_seed: int | None = Field(default=None, description="Optional deterministic seed")
    minutes_elapsed: int = Field(default=0, description="Elapsed simulated minutes")
    services: dict = Field(default_factory=dict, description="Detailed service state")
    alerts: list = Field(default_factory=list, description="Current active alerts")
    logs: list = Field(default_factory=list, description="Visible incident evidence")
    metrics: dict = Field(default_factory=dict, description="Per-service live metrics")
    timeline: list = Field(default_factory=list, description="Incident timeline")
    customer_impact: dict = Field(default_factory=dict, description="Customer impact summary")
    deploy_history: dict = Field(default_factory=dict, description="Deploy history by service")
    dependency_graph: dict = Field(default_factory=dict, description="Dependency notes by service")
    runbooks: dict = Field(default_factory=dict, description="Runbook snippets by keyword")
    action_history: list = Field(default_factory=list, description="Executed actions")
    hidden_causes: list = Field(default_factory=list, description="Hidden cause state for grading")
    analysis: dict = Field(default_factory=dict, description="Episode analysis counters")
    available_tools: list = Field(default_factory=list, description="Supported action grammar")
    last_action_result: dict = Field(default_factory=dict, description="Last action result")
