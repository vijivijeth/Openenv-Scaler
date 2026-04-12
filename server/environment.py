import copy
import json
import os
import random
import shlex
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import SREAction, SREObservation, SREState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SREResponseGymEnvironment(Environment):
    """Deterministic SRE incident-command simulator."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    TASK_ORDER = [
        "task_easy",
        "task_noisy_alert",
        "task_token_expiry",
        "task_medium",
        "task_hard",
        "task_pool_exhaustion",
        "task_expert",
        "task_trap",
        "task_extreme",
    ]
    AVAILABLE_TOOLS = [
        "check_logs <service>",
        "check_metrics <service>",
        "check_deploy <service>",
        "check_dependencies <service>",
        "query_runbook <keyword>",
        "restart_service <service>",
        "rollback_deploy <service> <version>",
        "scale_service <service> <replicas>",
        "drain_traffic <service>",
        "failover_db <cluster>",
        "isolate_az <zone>",
        "reset_circuit_breaker <service>",
        "acknowledge_alert <id>",
        "post_status <severity> <message>",
    ]
    TASK_CACHE: dict[str, dict] = {}
    HTTP_SHARED_SNAPSHOT: dict | None = None
    GLOBAL_EPISODE_HISTORY: list[dict] = []
    HISTORY_LIMIT = 100

    def __init__(self):
        super().__init__()
        self.tasks_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
        self.current_task: dict | None = None
        self.current_task_id = ""
        self.state_data: dict = {}
        self.step_count = 0
        self.minutes_elapsed = 0
        self._done = False
        self.action_history: list[str] = []
        self._episode_id = str(uuid4())
        self._scenario_seed: int | None = None
        self._revealed_sections: set[str] = set()
        self._unsafe_counts: dict[str, int] = {}
        self._state = SREState()
        self.analysis = self._new_analysis()

    @classmethod
    def _task_path(cls, task_id: str) -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "..", "tasks", f"{task_id}.json")

    @classmethod
    def load_task(cls, task_id: str) -> dict:
        if task_id not in cls.TASK_CACHE:
            with open(cls._task_path(task_id), encoding="utf-8") as f:
                cls.TASK_CACHE[task_id] = json.load(f)
        return copy.deepcopy(cls.TASK_CACHE[task_id])

    @classmethod
    def task_catalog(cls) -> list[dict]:
        return [cls.load_task(task_id) for task_id in cls.TASK_ORDER]

    @classmethod
    def history_summary(cls) -> dict:
        by_task: dict[str, list] = {}
        for episode in cls.GLOBAL_EPISODE_HISTORY:
            by_task.setdefault(episode["task_id"], []).append(episode)
        return {
            "total_episodes": len(cls.GLOBAL_EPISODE_HISTORY),
            "by_task": by_task,
            "recent": cls.GLOBAL_EPISODE_HISTORY[-10:],
        }

    @classmethod
    def snapshot_env(cls):
        if cls.HTTP_SHARED_SNAPSHOT is None:
            return None
        env = cls()
        env._load_runtime(cls.HTTP_SHARED_SNAPSHOT)
        return env

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SRE-ResponseGym",
            description=(
                "Deterministic SRE incident-command simulator with diagnostics, "
                "time progression, customer impact, and irreversible production risks."
            ),
            version="2.0.0",
            documentation_url="https://huggingface.co/spaces/vijivijeth/OpenEnv-SRE-ResponseGym",
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "task_easy",
        scenario_seed: int | None = None,
        **kwargs,
    ) -> SREObservation:
        self.current_task = self.load_task(task_id)
        self.current_task_id = task_id
        self._episode_id = episode_id or str(uuid4())
        self._scenario_seed = scenario_seed if scenario_seed is not None else seed
        self.state_data = copy.deepcopy(self.current_task["initial_state"])
        self.state_data.setdefault("logs", [])
        self.state_data.setdefault("alerts", [])
        self.state_data.setdefault("metrics", {})
        self.state_data.setdefault("deploy_history", {})
        self.state_data.setdefault("dependency_graph", {})
        self.state_data.setdefault("runbooks", {})
        self.state_data.setdefault("timeline", [])
        self.state_data.setdefault("customer_impact", {})
        self.state_data.setdefault("last_action_result", {})
        self.state_data.setdefault("available_tools", list(self.AVAILABLE_TOOLS))
        self.step_count = 0
        self.minutes_elapsed = 0
        self._done = False
        self.action_history = []
        self._revealed_sections = set()
        self._unsafe_counts = {}
        self.analysis = self._new_analysis()

        self._normalize_services()
        self._normalize_hidden_causes()
        self._apply_scenario_variation()
        self._append_timeline(
            self.current_task.get(
                "incident_commander_brief",
                f"Incident declared for {task_id}. Start triage immediately.",
            )
        )
        self._record_peak_impact()
        self._sync_state()
        self._save_http_snapshot()
        return self._build_observation(0.01)

    def step(
        self,
        action: SREAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> SREObservation:
        self._load_http_snapshot_if_needed()
        if not self.current_task:
            return self._build_observation(0.01)
        if self._done:
            return self._build_observation(0.01)

        self.step_count += 1
        self.minutes_elapsed += 5

        action_str = action.action.strip().lower()
        self.action_history.append(action_str)
        parsed = self._parse_action(action_str)
        reward = self._execute_action(action_str, parsed)

        self._progress_incident()
        self._update_dependency_health()
        self._record_peak_impact()

        if self.step_count >= self.current_task["max_steps"] or self._check_done():
            self._done = True

        if self._done:
            self._append_timeline("Incident closed. Benchmark run completed.")
        self._sync_state()
        self._save_http_snapshot()
        if self._done:
            self._record_episode()

        return self._build_observation(reward)

    @property
    def state(self) -> SREState:
        self._load_http_snapshot_if_needed()
        return self._state

    def current_observation(self) -> SREObservation:
        self._load_http_snapshot_if_needed()
        return self._build_observation(0.01)

    def get_grade(self) -> dict:
        if not self.current_task:
            return {
                "final_score": 0.50,
                "letter_grade": "C",
                "breakdown": {
                    "resolution": 0.50,
                    "diagnosis_accuracy": 0.50,
                    "operational_safety": 0.50,
                    "efficiency": 0.50,
                    "incident_hygiene": 0.50,
                },
                "analysis": {},
                "actions_taken": [],
                "steps_used": 0,
                "max_steps": 0,
            }
        from server.grader import Grader

        grader = Grader(self.current_task)
        return grader.grade(
            final_services=self.service_statuses,
            action_history=self.action_history,
            step_count=self.step_count,
            analysis_state=self.analysis,
            customer_impact=self.state_data.get("customer_impact", {}),
        )

    @property
    def service_statuses(self) -> dict:
        return {
            service: details.get("status", "unknown")
            for service, details in self.state_data.get("services", {}).items()
        }

    def _new_analysis(self) -> dict:
        return {
            "acknowledged_alert_ids": [],
            "status_updates": [],
            "redundant_actions": 0,
            "forbidden_action_hits": 0,
            "irreversible_mistakes": 0,
            "diagnosis_actions_taken": [],
            "time_to_first_diagnosis": None,
            "unsafe_events": [],
            "peak_users_affected_pct": 0.0,
        }

    def _normalize_services(self) -> None:
        for details in self.state_data.get("services", {}).values():
            details.setdefault("dependencies", [])
            details.setdefault("az", "global")
            details.setdefault("current_version", "unknown")
            details.setdefault("stable_version", details.get("current_version", "unknown"))
            details.setdefault("replicas", 1)
            details.setdefault("desired_replicas", details.get("replicas", 1))
            details.setdefault("traffic", "live")
            details.setdefault("auto_recover", False)

    def _normalize_hidden_causes(self) -> None:
        for cause in self.current_task.get("hidden_causes", []):
            cause.setdefault("resolved", False)
            cause.setdefault("affected_services", [cause.get("service")] if cause.get("service") else [])
            cause.setdefault("evidence_actions", [])
            cause.setdefault("prerequisite_actions", [])
            cause.setdefault("unsafe_actions", [])
            cause.setdefault("impact", 0.15)

    def _apply_scenario_variation(self) -> None:
        if self._scenario_seed is None:
            return
        rng = random.Random(self._scenario_seed)
        logs = list(self.state_data.get("logs", []))
        if len(logs) > 1:
            rotate = rng.randint(0, len(logs) - 1)
            self.state_data["logs"] = logs[rotate:] + logs[:rotate]

    def _load_http_snapshot_if_needed(self) -> None:
        if self.current_task is None and self.HTTP_SHARED_SNAPSHOT is not None:
            self._load_runtime(self.HTTP_SHARED_SNAPSHOT)

    def _save_http_snapshot(self) -> None:
        type(self).HTTP_SHARED_SNAPSHOT = self._serialize_runtime()

    def _serialize_runtime(self) -> dict:
        return {
            "current_task": copy.deepcopy(self.current_task),
            "current_task_id": self.current_task_id,
            "state_data": copy.deepcopy(self.state_data),
            "step_count": self.step_count,
            "minutes_elapsed": self.minutes_elapsed,
            "done": self._done,
            "action_history": list(self.action_history),
            "episode_id": self._episode_id,
            "scenario_seed": self._scenario_seed,
            "revealed_sections": list(self._revealed_sections),
            "unsafe_counts": copy.deepcopy(self._unsafe_counts),
            "analysis": copy.deepcopy(self.analysis),
        }

    def _load_runtime(self, snapshot: dict) -> None:
        self.current_task = copy.deepcopy(snapshot["current_task"])
        self.current_task_id = snapshot["current_task_id"]
        self.state_data = copy.deepcopy(snapshot["state_data"])
        self.step_count = snapshot["step_count"]
        self.minutes_elapsed = snapshot["minutes_elapsed"]
        self._done = snapshot["done"]
        self.action_history = list(snapshot["action_history"])
        self._episode_id = snapshot["episode_id"]
        self._scenario_seed = snapshot["scenario_seed"]
        self._revealed_sections = set(snapshot["revealed_sections"])
        self._unsafe_counts = copy.deepcopy(snapshot["unsafe_counts"])
        self.analysis = copy.deepcopy(snapshot["analysis"])
        self._sync_state()

    def _sync_state(self) -> None:
        self._state = SREState(
            episode_id=self._episode_id,
            task_id=self.current_task_id,
            step_count=self.step_count,
            done=self._done,
            scenario_seed=self._scenario_seed,
            minutes_elapsed=self.minutes_elapsed,
            services=copy.deepcopy(self.state_data.get("services", {})),
            alerts=copy.deepcopy(self.state_data.get("alerts", [])),
            logs=list(self.state_data.get("logs", [])[-10:]),
            metrics=copy.deepcopy(self.state_data.get("metrics", {})),
            timeline=list(self.state_data.get("timeline", [])[-10:]),
            customer_impact=copy.deepcopy(self.state_data.get("customer_impact", {})),
            deploy_history=copy.deepcopy(self.state_data.get("deploy_history", {})),
            dependency_graph=copy.deepcopy(self.state_data.get("dependency_graph", {})),
            runbooks=copy.deepcopy(self.state_data.get("runbooks", {})),
            action_history=list(self.action_history),
            hidden_causes=copy.deepcopy(self.current_task.get("hidden_causes", []) if self.current_task else []),
            analysis=copy.deepcopy(self.analysis),
            available_tools=list(self.AVAILABLE_TOOLS),
            last_action_result=copy.deepcopy(self.state_data.get("last_action_result", {})),
        )

    def _build_observation(self, reward: float) -> SREObservation:
        if not self.current_task:
            return SREObservation(
                task_id="",
                description="No task loaded. Call reset() first.",
                services={},
                alerts=[],
                logs=[],
                metrics={},
                timeline=[],
                customer_impact={},
                available_tools=list(self.AVAILABLE_TOOLS),
                last_action_result={},
                step=0,
                max_steps=0,
                reward=0.01,
                done=False,
            )

        return SREObservation(
            task_id=self.current_task_id,
            description=self.current_task.get("description", ""),
            services=self.service_statuses,
            alerts=copy.deepcopy(self.state_data.get("alerts", [])),
            logs=list(self.state_data.get("logs", [])[-10:]),
            metrics=copy.deepcopy(self.state_data.get("metrics", {})),
            timeline=list(self.state_data.get("timeline", [])[-8:]),
            customer_impact=copy.deepcopy(self.state_data.get("customer_impact", {})),
            available_tools=list(self.AVAILABLE_TOOLS),
            last_action_result=copy.deepcopy(self.state_data.get("last_action_result", {})),
            step=self.step_count,
            max_steps=self.current_task.get("max_steps", 0),
            reward=round(_clamp(float(reward), 0.01, 0.99), 3),
            done=self._done,
        )

    def _parse_action(self, action: str) -> dict:
        try:
            parts = shlex.split(action)
        except ValueError:
            return {"valid": False, "command": "", "args": [], "error": "Invalid shell-style quoting."}

        if not parts:
            return {"valid": False, "command": "", "args": [], "error": "Action cannot be empty."}

        command = parts[0]
        args = parts[1:]
        requirements = {
            "check_logs": 1,
            "check_metrics": 1,
            "check_deploy": 1,
            "check_dependencies": 1,
            "query_runbook": 1,
            "restart_service": 1,
            "rollback_deploy": 2,
            "scale_service": 2,
            "drain_traffic": 1,
            "failover_db": 1,
            "isolate_az": 1,
            "reset_circuit_breaker": 1,
            "acknowledge_alert": 1,
            "post_status": 2,
        }
        minimum_args = requirements.get(command)
        if minimum_args is None:
            return {
                "valid": False,
                "command": command,
                "args": args,
                "error": f"Unsupported action '{command}'.",
            }
        if len(args) < minimum_args:
            return {
                "valid": False,
                "command": command,
                "args": args,
                "error": f"Action '{command}' is missing required arguments.",
            }
        return {"valid": True, "command": command, "args": args, "error": None}

    def _execute_action(self, action: str, parsed: dict) -> float:
        if not parsed["valid"]:
            self.analysis["redundant_actions"] += 1
            self._set_action_result(
                parsed.get("command", "invalid"),
                "",
                "invalid",
                parsed["error"],
            )
            return 0.01

        if self._apply_unsafe_action(action):
            return 0.01

        command = parsed["command"]
        args = parsed["args"]

        if command == "check_logs":
            return self._handle_check_logs(args[0], action)
        if command == "check_metrics":
            return self._handle_check_metrics(args[0], action)
        if command == "check_deploy":
            return self._handle_check_deploy(args[0], action)
        if command == "check_dependencies":
            return self._handle_check_dependencies(args[0], action)
        if command == "query_runbook":
            return self._handle_query_runbook(args[0], action)
        if command == "restart_service":
            return self._handle_restart(args[0], action)
        if command == "rollback_deploy":
            return self._handle_rollback(args[0], args[1], action)
        if command == "scale_service":
            return self._handle_scale(args[0], args[1], action)
        if command == "drain_traffic":
            return self._handle_drain(args[0], action)
        if command == "failover_db":
            return self._handle_failover(args[0], action)
        if command == "isolate_az":
            return self._handle_isolate(args[0], action)
        if command == "reset_circuit_breaker":
            return self._handle_circuit_reset(args[0], action)
        if command == "acknowledge_alert":
            return self._handle_acknowledge(args[0], action)
        if command == "post_status":
            return self._handle_post_status(args[0], " ".join(args[1:]), action)

        self.analysis["redundant_actions"] += 1
        self._set_action_result(command, "", "noop", "Action had no effect.")
        return 0.01

    def _handle_check_logs(self, service: str, action: str) -> float:
        lines = self.current_task["initial_state"].get("evidence", {}).get("logs", {}).get(service, [])
        return self._reveal_section(
            action,
            f"logs:{service}",
            lines or [f"{service}: no additional log lines available."],
            "diagnostic",
            service,
        )

    def _handle_check_metrics(self, service: str, action: str) -> float:
        metrics = self.state_data.get("metrics", {}).get(service)
        if metrics:
            summary = ", ".join(f"{key}={value}" for key, value in metrics.items())
            lines = [f"{service} metrics: {summary}"]
        else:
            lines = [f"{service}: no metrics available."]
        return self._reveal_section(action, f"metrics:{service}", lines, "diagnostic", service)

    def _handle_check_deploy(self, service: str, action: str) -> float:
        lines = self.state_data.get("deploy_history", {}).get(service, [])
        return self._reveal_section(
            action,
            f"deploy:{service}",
            lines or [f"{service}: no deploy history found."],
            "diagnostic",
            service,
        )

    def _handle_check_dependencies(self, service: str, action: str) -> float:
        lines = self.state_data.get("dependency_graph", {}).get(service, [])
        if not lines:
            dependencies = self.state_data["services"].get(service, {}).get("dependencies", [])
            lines = [f"{service} dependencies: {', '.join(dependencies) if dependencies else 'none'}"]
        return self._reveal_section(action, f"deps:{service}", lines, "diagnostic", service)

    def _handle_query_runbook(self, keyword: str, action: str) -> float:
        runbooks = self.state_data.get("runbooks", {})
        match_key = None
        if keyword in runbooks:
            match_key = keyword
        else:
            for key in runbooks:
                if keyword in key:
                    match_key = key
                    break
        if match_key is None:
            self.analysis["redundant_actions"] += 1
            self.state_data["logs"].append(f"[RUNBOOK] no entry found for {keyword}")
            self._set_action_result("query_runbook", keyword, "noop", "No runbook entry found.")
            return 0.02
        return self._reveal_section(
            action,
            f"runbook:{match_key}",
            [f"[RUNBOOK] {match_key}: {runbooks[match_key]}"],
            "diagnostic",
            match_key,
        )

    def _handle_restart(self, service: str, action: str) -> float:
        service_state = self.state_data["services"].get(service)
        if service_state is None:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("restart_service", service, "invalid", "Unknown service.")
            return 0.01

        resolved = self._resolve_matching_causes(action)
        if resolved:
            service_state["status"] = "healthy"
            service_state["traffic"] = "live"
            self._append_log(f"[SYSTEM] {service} restarted after mitigation and is now healthy.")
            self._set_action_result("restart_service", service, "resolved", resolved[0]["summary"])
            return 0.24

        if self._service_has_blockers(service):
            service_state["status"] = "degraded"
            self.analysis["redundant_actions"] += 1
            detail = f"{service} restarted but root cause or dependency blockers remain."
            self._append_log(f"[SYSTEM] {detail}")
            self._set_action_result("restart_service", service, "blocked", detail)
            return 0.02

        if service_state["status"] != "healthy":
            service_state["status"] = "healthy"
            service_state["traffic"] = "live"
            self._append_log(f"[SYSTEM] {service} restarted cleanly.")
            self._set_action_result("restart_service", service, "recovered", "Service is healthy.")
            return 0.18

        self.analysis["redundant_actions"] += 1
        self._set_action_result("restart_service", service, "noop", "Service was already healthy.")
        return 0.02

    def _handle_rollback(self, service: str, version: str, action: str) -> float:
        resolved = self._resolve_matching_causes(action)
        if resolved:
            service_state = self.state_data["services"][service]
            service_state["current_version"] = version
            service_state["status"] = "healthy"
            self._append_log(f"[SYSTEM] {service} rolled back to {version}.")
            self._set_action_result("rollback_deploy", service, "resolved", resolved[0]["summary"])
            return 0.32

        self.analysis["redundant_actions"] += 1
        self._set_action_result("rollback_deploy", service, "noop", "Rollback target did not match the active incident.")
        return 0.02

    def _handle_scale(self, service: str, replicas: str, action: str) -> float:
        try:
            replica_count = int(replicas)
        except ValueError:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("scale_service", service, "invalid", "Replica count must be an integer.")
            return 0.01

        service_state = self.state_data["services"].get(service)
        if service_state is None:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("scale_service", service, "invalid", "Unknown service.")
            return 0.01

        service_state["replicas"] = replica_count
        resolved = self._resolve_matching_causes(action)
        if resolved:
            service_state["status"] = "healthy"
            self._append_log(f"[SYSTEM] {service} scaled to {replica_count} replicas and stabilized.")
            self._set_action_result("scale_service", service, "resolved", resolved[0]["summary"])
            return 0.26

        self._append_log(f"[SYSTEM] {service} scaled to {replica_count} replicas.")
        self._set_action_result("scale_service", service, "partial", "Capacity changed but no cause was fully resolved.")
        return 0.12

    def _handle_drain(self, service: str, action: str) -> float:
        service_state = self.state_data["services"].get(service)
        if service_state is None:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("drain_traffic", service, "invalid", "Unknown service.")
            return 0.01

        if service_state.get("traffic") == "drained":
            self.analysis["redundant_actions"] += 1
            self._set_action_result("drain_traffic", service, "noop", "Traffic was already drained.")
            return 0.02

        service_state["traffic"] = "drained"
        resolved = self._resolve_matching_causes(action)
        detail = "Traffic drained to reduce customer impact."
        if resolved:
            detail = resolved[0]["summary"]
        self._append_log(f"[SYSTEM] {service} traffic drained.")
        self._set_action_result("drain_traffic", service, "mitigated", detail)
        return 0.18

    def _handle_failover(self, cluster: str, action: str) -> float:
        resolved = self._resolve_matching_causes(action)
        if resolved:
            if cluster in self.state_data["services"]:
                self.state_data["services"][cluster]["status"] = "healthy"
            self._append_log(f"[SYSTEM] {cluster} failed over successfully.")
            self._set_action_result("failover_db", cluster, "resolved", resolved[0]["summary"])
            return 0.28

        self.analysis["redundant_actions"] += 1
        self._set_action_result("failover_db", cluster, "noop", "Failover preconditions were not met.")
        return 0.02

    def _handle_isolate(self, zone: str, action: str) -> float:
        resolved = self._resolve_matching_causes(action)
        if resolved:
            self._append_log(f"[SYSTEM] Availability zone {zone} fenced from the incident.")
            self._set_action_result("isolate_az", zone, "resolved", resolved[0]["summary"])
            return 0.28

        self.analysis["redundant_actions"] += 1
        self._set_action_result("isolate_az", zone, "noop", "Isolation did not match an active partition.")
        return 0.02

    def _handle_circuit_reset(self, service: str, action: str) -> float:
        resolved = self._resolve_matching_causes(action)
        if resolved:
            self.state_data["services"][service]["status"] = "healthy"
            self._append_log(f"[SYSTEM] Circuit breaker on {service} reset. Traffic is resuming.")
            self._set_action_result("reset_circuit_breaker", service, "resolved", resolved[0]["summary"])
            return 0.28

        blockers = self._service_has_blockers(service)
        if blockers:
            self.state_data["services"][service]["status"] = "degraded"
            self.analysis["forbidden_action_hits"] += 1
            detail = f"{service} breaker re-tripped because upstream recovery was incomplete."
            self._append_log(f"[SYSTEM] {detail}")
            self._set_action_result("reset_circuit_breaker", service, "unsafe", detail)
            return 0.01

        self.state_data["services"][service]["status"] = "healthy"
        self._set_action_result("reset_circuit_breaker", service, "recovered", "Traffic resumed successfully.")
        return 0.18

    def _handle_acknowledge(self, alert_id: str, action: str) -> float:
        try:
            alert_id_value = int(alert_id)
        except ValueError:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("acknowledge_alert", alert_id, "invalid", "Alert ID must be numeric.")
            return 0.01

        alerts = self.state_data.get("alerts", [])
        remaining = [alert for alert in alerts if alert.get("id") != alert_id_value]
        if len(remaining) == len(alerts):
            self.analysis["redundant_actions"] += 1
            self._set_action_result("acknowledge_alert", str(alert_id_value), "noop", "Alert not found.")
            return 0.02

        self.state_data["alerts"] = remaining
        if alert_id_value not in self.analysis["acknowledged_alert_ids"]:
            self.analysis["acknowledged_alert_ids"].append(alert_id_value)
        self._append_timeline(f"Critical alert {alert_id_value} acknowledged by the incident commander.")
        self._set_action_result("acknowledge_alert", str(alert_id_value), "tracked", "Alert acknowledged.")
        return 0.05

    def _handle_post_status(self, severity: str, message: str, action: str) -> float:
        if severity not in {"info", "minor", "major", "critical"}:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("post_status", severity, "invalid", "Unsupported status severity.")
            return 0.01

        clean_message = message.strip()
        if len(clean_message) < 8:
            self.analysis["redundant_actions"] += 1
            self._set_action_result("post_status", severity, "invalid", "Status message is too short.")
            return 0.01

        self.analysis["status_updates"].append({"severity": severity, "message": clean_message})
        self._append_timeline(f"Status update ({severity}): {clean_message}")
        self._set_action_result("post_status", severity, "communicated", "Status update published.")
        return 0.04

    def _reveal_section(self, action: str, section_key: str, lines: list[str], outcome: str, target: str) -> float:
        if section_key in self._revealed_sections:
            self.analysis["redundant_actions"] += 1
            self._set_action_result(section_key.split(":")[0], target, "noop", "Evidence already reviewed.")
            return 0.02

        self._revealed_sections.add(section_key)
        for line in lines:
            self._append_log(line)

        if action in self.current_task.get("diagnosis_actions", []):
            if action not in self.analysis["diagnosis_actions_taken"]:
                self.analysis["diagnosis_actions_taken"].append(action)
            if self.analysis["time_to_first_diagnosis"] is None:
                self.analysis["time_to_first_diagnosis"] = self.step_count
            self._set_action_result(section_key.split(":")[0], target, outcome, "Root-cause evidence gathered.")
            return 0.08

        self._set_action_result(section_key.split(":")[0], target, outcome, "Additional evidence reviewed.")
        return 0.05

    def _resolve_matching_causes(self, action: str) -> list[dict]:
        resolved = []
        for cause in self.current_task.get("hidden_causes", []):
            if cause.get("resolved"):
                continue
            if action != cause.get("fix_action", ""):
                continue
            if not self._prerequisites_satisfied(cause):
                continue
            cause["resolved"] = True
            resolved.append(cause)
            for service in cause.get("affected_services", []):
                if service in self.state_data["services"]:
                    self.state_data["services"][service]["status"] = "healthy"
            if cause.get("service") in self.state_data["services"]:
                service_state = self.state_data["services"][cause["service"]]
                service_state["status"] = "healthy"
                if cause.get("type") in {"bad_deploy", "connection_leak"}:
                    stable = service_state.get("stable_version", service_state["current_version"])
                    service_state["current_version"] = stable
            if cause.get("type") == "scale_required" and cause.get("service") in self.state_data["services"]:
                target = self.state_data["services"][cause["service"]]["desired_replicas"]
                self.state_data["services"][cause["service"]]["replicas"] = target
            if cause.get("type") == "traffic_reload" and cause.get("service") in self.state_data["services"]:
                self.state_data["services"][cause["service"]]["traffic"] = "live"
            self._append_timeline(cause.get("resolution_log", f"Resolved cause: {cause.get('summary', cause['id'])}"))
        return resolved

    def _prerequisites_satisfied(self, cause: dict) -> bool:
        prerequisites = cause.get("prerequisite_actions", [])
        return all(prerequisite in self.action_history for prerequisite in prerequisites)

    def _apply_unsafe_action(self, action: str) -> bool:
        for cause in self.current_task.get("hidden_causes", []):
            if cause.get("resolved"):
                continue
            for unsafe in cause.get("unsafe_actions", []):
                if action != unsafe.get("action"):
                    continue
                if all(prereq in self.action_history[:-1] for prereq in unsafe.get("safe_after", [])):
                    continue
                service = unsafe.get("service", cause.get("service", ""))
                if service in self.state_data.get("services", {}):
                    status = unsafe.get("resulting_status", "degraded")
                    self.state_data["services"][service]["status"] = status
                key = unsafe["action"]
                self._unsafe_counts[key] = self._unsafe_counts.get(key, 0) + 1
                if unsafe.get("lock_after") and service in self.state_data["services"]:
                    if self._unsafe_counts[key] >= unsafe["lock_after"]:
                        self.state_data["services"][service]["status"] = "locked"
                if unsafe.get("irreversible"):
                    self.analysis["irreversible_mistakes"] += 1
                self.analysis["forbidden_action_hits"] += 1
                self.analysis["unsafe_events"].append(unsafe.get("consequence", "unsafe action executed"))
                detail = unsafe.get("consequence", "Unsafe action made the incident worse.")
                self._append_log(f"[SYSTEM] {detail}")
                self._append_timeline(detail)
                self._set_action_result(action.split()[0], service, "unsafe", detail)
                return True
        return False

    def _service_has_blockers(self, service: str) -> bool:
        for cause in self.current_task.get("hidden_causes", []):
            if cause.get("resolved"):
                continue
            if service in cause.get("affected_services", []):
                if cause.get("service") == service:
                    return True
        dependencies = self.state_data["services"].get(service, {}).get("dependencies", [])
        for dependency in dependencies:
            if self.state_data["services"].get(dependency, {}).get("status") != "healthy":
                return True
        return False

    def _update_dependency_health(self) -> None:
        for service, details in self.state_data.get("services", {}).items():
            dependencies = details.get("dependencies", [])
            dependency_unhealthy = any(
                self.state_data["services"].get(dep, {}).get("status") != "healthy"
                for dep in dependencies
            )
            if dependency_unhealthy and details.get("status") == "healthy" and details.get("traffic") != "drained":
                details["status"] = "degraded"

            if (
                details.get("auto_recover")
                and not dependency_unhealthy
                and not self._service_has_blockers(service)
                and details.get("status") in {"degraded", "down"}
            ):
                details["status"] = "healthy"

    def _progress_incident(self) -> None:
        unresolved = [cause for cause in self.current_task.get("hidden_causes", []) if not cause.get("resolved")]
        impact_score = sum(float(cause.get("impact", 0.1)) for cause in unresolved)
        drained_services = sum(
            1
            for service in self.state_data.get("services", {}).values()
            if service.get("traffic") == "drained"
        )
        scaled_capacity_bonus = sum(
            max(0, details.get("replicas", 1) - details.get("desired_replicas", 1))
            for details in self.state_data.get("services", {}).values()
        )

        customer = self.state_data.get("customer_impact", {})
        base_users = float(customer.get("users_affected_pct", 10.0))
        if unresolved:
            users = base_users + impact_score * 8.0 + self.step_count * 1.4 - drained_services * 5.0 - scaled_capacity_bonus * 2.0
        else:
            users = max(1.0, base_users - 10.0)
        users = _clamp(users, 1.0, 99.0)

        customer["users_affected_pct"] = round(users, 1)
        customer["revenue_risk_per_min"] = int(_clamp(users * 85, 80, 12000))
        customer["error_budget_burn_rate"] = round(_clamp(0.8 + impact_score * 6.5 - drained_services * 0.7, 0.3, 18.0), 2)
        customer["minutes_elapsed"] = self.minutes_elapsed
        customer["severity"] = (
            "sev1" if users >= 60 else
            "sev2" if users >= 35 else
            "sev3" if users >= 15 else "sev4"
        )

        for service, metrics in self.state_data.get("metrics", {}).items():
            status = self.state_data["services"].get(service, {}).get("status")
            if status in {"down", "degraded", "unreachable", "locked"}:
                metrics["error_rate_pct"] = round(_clamp(float(metrics.get("error_rate_pct", 5.0)) + 2.5, 0, 100), 1)
                metrics["latency_ms"] = int(_clamp(float(metrics.get("latency_ms", 120)) + 45, 10, 5000))
                if "queue_depth" in metrics:
                    metrics["queue_depth"] = int(_clamp(float(metrics.get("queue_depth", 0)) + 18 - scaled_capacity_bonus * 6, 0, 5000))
                if "memory_pct" in metrics:
                    metrics["memory_pct"] = round(_clamp(float(metrics.get("memory_pct", 40)) + 4, 0, 100), 1)
                if "connections_used" in metrics and "connections_limit" in metrics:
                    metrics["connections_used"] = int(_clamp(float(metrics.get("connections_used", 0)) + 8, 0, float(metrics.get("connections_limit", 100))))
                if "replica_lag_ms" in metrics:
                    metrics["replica_lag_ms"] = int(_clamp(float(metrics.get("replica_lag_ms", 0)) + 40, 0, 5000))
            else:
                metrics["error_rate_pct"] = round(_clamp(float(metrics.get("error_rate_pct", 1.0)) - 1.2, 0, 100), 1)
                metrics["latency_ms"] = int(_clamp(float(metrics.get("latency_ms", 120)) - 25, 10, 5000))
                if "queue_depth" in metrics:
                    metrics["queue_depth"] = int(_clamp(float(metrics.get("queue_depth", 0)) - 14 - scaled_capacity_bonus * 4, 0, 5000))
                if "memory_pct" in metrics:
                    metrics["memory_pct"] = round(_clamp(float(metrics.get("memory_pct", 40)) - 2.5, 0, 100), 1)
                if "connections_used" in metrics and "connections_limit" in metrics:
                    metrics["connections_used"] = int(_clamp(float(metrics.get("connections_used", 0)) - 5, 0, float(metrics.get("connections_limit", 100))))
                if "replica_lag_ms" in metrics:
                    metrics["replica_lag_ms"] = int(_clamp(float(metrics.get("replica_lag_ms", 0)) - 35, 0, 5000))

    def _record_peak_impact(self) -> None:
        users = float(self.state_data.get("customer_impact", {}).get("users_affected_pct", 0.0))
        self.analysis["peak_users_affected_pct"] = max(float(self.analysis.get("peak_users_affected_pct", 0.0)), users)

    def _check_done(self) -> bool:
        success = self.current_task.get("success_criteria", {})
        required_services = success.get("required_services", [])
        required_actions = success.get("required_actions", [])
        required_status = success.get("required_status", "healthy")
        services_ready = all(
            self.state_data["services"].get(service, {}).get("status") == required_status
            for service in required_services
        )
        actions_ready = all(action in self.action_history for action in required_actions)
        return services_ready and actions_ready

    def _record_episode(self) -> None:
        grade = self.get_grade()
        record = {
            "episode_id": self._episode_id,
            "task_id": self.current_task_id,
            "score": grade["final_score"],
            "steps": self.step_count,
            "breakdown": grade["breakdown"],
            "customer_impact": copy.deepcopy(self.state_data.get("customer_impact", {})),
        }
        type(self).GLOBAL_EPISODE_HISTORY.append(record)
        if len(type(self).GLOBAL_EPISODE_HISTORY) > self.HISTORY_LIMIT:
            type(self).GLOBAL_EPISODE_HISTORY = type(self).GLOBAL_EPISODE_HISTORY[-self.HISTORY_LIMIT:]

    def _append_log(self, line: str) -> None:
        self.state_data.setdefault("logs", []).append(line)

    def _append_timeline(self, detail: str) -> None:
        stamp = f"[+{self.minutes_elapsed:02d}m]"
        self.state_data.setdefault("timeline", []).append(f"{stamp} {detail}")

    def _set_action_result(self, command: str, target: str, outcome: str, detail: str) -> None:
        self.state_data["last_action_result"] = {
            "command": command,
            "target": target,
            "outcome": outcome,
            "detail": detail,
            "minutes_elapsed": self.minutes_elapsed,
        }
        if detail:
            prefix = f"{command} {target}".strip()
            self._append_timeline(f"{prefix} -> {detail}")
