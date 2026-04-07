import json
import os
from server.grader import Grader


class IncidentEnv:

    def __init__(self):
        self.episode_history = []  # stores all completed episodes
        self.current_episode_id = 0
        self.current_task = None
        self.state_data = None
        self.step_count = 0
        self.done = False
        self.wrong_restarts = 0
        self.action_history = []
        self.tasks_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, task_id: str = "task_easy") -> dict:
        task_file = os.path.join(self.tasks_dir, f"{task_id}.json")

        with open(task_file, "r") as f:
            self.current_task = json.load(f)

        self.state_data = json.loads(
            json.dumps(self.current_task["initial_state"])
        )

        self.step_count = 0
        self.done = False
        self.wrong_restarts = 0
        self.action_history = []

        return self._build_observation()

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: str) -> dict:
        if self.done:
            return {
                "observation": self._build_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already finished. Call reset()."}
            }

        self.step_count += 1
        self.action_history.append(action)

        reward = self._apply_action(action)
        self.done = self._check_done()

        if self.step_count >= self.current_task["max_steps"]:
            self.done = True

        if self.done:
            try:
                grader = Grader(self.current_task)
                grade = grader.grade(
                    final_services=self.state_data["services"],
                    action_history=self.action_history,
                    step_count=self.step_count
                )
                self.episode_history.append({
                    "episode_id": self.current_episode_id,
                    "task_id": self.current_task["task_id"],
                    "score": grade["final_score"],
                    "steps": self.step_count,
                    "actions": list(self.action_history),
                    "breakdown": grade["breakdown"]
                })
                self.current_episode_id += 1
                # Keep only last 50 episodes
                if len(self.episode_history) > 50:
                    self.episode_history = self.episode_history[-50:]
            except Exception:
                pass

        return {
            "observation": self._build_observation(),
            "reward": reward,
            "done": self.done,
            "info": {
                "step": self.step_count,
                "action_taken": action,
                "action_history": self.action_history
            }
        }

    # -------------------------
    # STATE
    # -------------------------
    def state(self) -> dict:
        return {
            "task_id": self.current_task["task_id"],
            "step_count": self.step_count,
            "done": self.done,
            "services": self.state_data["services"],
            "alerts": self.state_data["alerts"],
            "logs": self.state_data["logs"]
        }

    # -------------------------
    # APPLY ACTION
    # -------------------------
    def _apply_action(self, action: str) -> float:
        action = action.strip().lower()

        # --- restart_service ---
        if action.startswith("restart_service"):
            return self._handle_restart(action)

        # --- check_logs ---
        if action.startswith("check_logs"):
            parts = action.split()
            service = parts[1] if len(parts) > 1 else ""
            self.state_data["logs"].append(
                f"[agent checked logs for: {service}]"
            )
            return 0.05

        # --- query_runbook ---
        if action.startswith("query_runbook"):
            parts = action.split()
            keyword = parts[1] if len(parts) > 1 else ""
            
            # Check both locations — inside initial_state and at task root
            runbook = (
                self.current_task.get("initial_state", {}).get("runbook", {})
                or self.current_task.get("runbook", {})
            )
            
            if keyword in runbook:
                self.state_data["logs"].append(
                    f"[RUNBOOK] {keyword}: {runbook[keyword]}"
                )
                return 0.1
            
            # Partial match — keyword appears inside a key
            for key, val in runbook.items():
                if keyword.lower() in key.lower():
                    self.state_data["logs"].append(
                        f"[RUNBOOK] {key}: {val}"
                    )
                    return 0.1
            
            self.state_data["logs"].append(
                f"[RUNBOOK] No entry found for: {keyword}"
            )
            return 0.0

        # --- rollback_deploy ---
        if action.startswith("rollback_deploy"):
            return self._handle_rollback(action)
        
        # --- isolate_az ---
        if action.startswith("isolate_az"):
            parts = action.split()
            az = parts[1] if len(parts) > 1 else ""
            self.state_data["logs"].append(
                f"[SYSTEM] Availability zone {az} isolated successfully."
                " Split-brain risk eliminated."
            )
            if "database-primary" in self.state_data["services"]:
                self.state_data["services"]["database-primary"] = "healthy"
            return 0.3

        # --- failover_db ---
        if action.startswith("failover_db"):
            parts = action.split()
            db = parts[1] if len(parts) > 1 else ""
            required = self.current_task.get(
                "win_condition", {}
            ).get("required_action_taken", "")
            if action == required or "failover_db" in required:
                self.state_data["services"][db] = "healthy"
                self.state_data["logs"].append(
                    f"[SYSTEM] {db} failed over to replica successfully."
                    " Replica promoted to primary."
                )
                return 0.4
            return 0.0

        # --- reset_circuit_breaker ---
        if action.startswith("reset_circuit_breaker"):
            parts = action.split()
            service = parts[1] if len(parts) > 1 else ""
            required = self.current_task.get(
                "win_condition", {}
            ).get("required_action_taken", "")
            previous_resets = [
                a for a in self.action_history[:-1]
                if a.startswith("reset_circuit_breaker")
            ]
            if previous_resets:
                return 0.0
            if service in self.state_data["services"]:
                self.state_data["services"][service] = "healthy"
                self.state_data["logs"].append(
                    f"[SYSTEM] Circuit breaker on {service} reset successfully."
                    " Traffic resuming."
                )
                return 0.5
            return 0.0

        # --- acknowledge_alert ---
        if action.startswith("acknowledge_alert"):
            parts = action.split()
            alert_id = int(parts[1]) if len(parts) > 1 else -1
            self.state_data["alerts"] = [
                a for a in self.state_data["alerts"]
                if a.get("id") != alert_id
            ]
            return 0.05

        return 0.0

    # -------------------------
    # HANDLE RESTART
    # -------------------------
    def _handle_restart(self, action: str) -> float:
        parts = action.split()
        service = parts[1] if len(parts) > 1 else ""
        trap = self.current_task.get("trap", {})

        # Hard task trap — restarting a service stuck in crash loop
        if action == trap.get("wrong_action", ""):
            already_isolated = any(
                a.startswith("isolate_az") for a in self.action_history[:-1]
            )
            if already_isolated:
                if service in self.state_data["services"]:
                    self.state_data["services"][service] = "healthy"
                    self.state_data["logs"].append(
                        f"[SYSTEM] {service} restarted safely after AZ isolation."
                    )
                    return 0.3
            self.wrong_restarts += 1
            max_wrong = trap.get("max_wrong_restarts", 999)
            if self.wrong_restarts >= max_wrong:
                self.state_data["services"][service] = "locked"
                self.state_data["logs"].append(
                    f"[SYSTEM] {service} locked after too many failed attempts."
                )
                return trap.get("penalty", -0.3)
            self.state_data["services"][service] = "degraded"
            self.state_data["logs"].append(
                f"[SYSTEM] {service} {trap.get('consequence','failed again')}"
            )
            return trap.get("penalty", -0.3)

        # Medium task trap — fixing the wrong service first
        medium_trap = self.current_task.get("trap", {})
        if (action == medium_trap.get("wrong_first_action", "")
                and self.step_count <= 2):
            self.state_data["services"][service] = "degraded"
            self.state_data["logs"].append(
                f"[SYSTEM] {service} restarted but immediately degraded again"
                " — root cause is still unresolved."
            )
            return medium_trap.get("penalty", -0.3)

        # Correct restart
        if service in self.state_data["services"]:
            self.state_data["services"][service] = "healthy"
            self.state_data["logs"].append(
                f"[SYSTEM] {service} restarted successfully."
            )
            self.state_data["alerts"] = [
                a for a in self.state_data["alerts"]
                if service not in a.get("message", "")
            ]
            return 0.3

        return 0.0

    # -------------------------
    # HANDLE ROLLBACK
    # -------------------------
    def _handle_rollback(self, action: str) -> float:
        parts = action.split()
        if len(parts) < 3:
            return 0.0

        service = parts[1]
        version = parts[2]

        correct_actions = self.current_task.get("solution_actions", [])

        # Only reward rollback once
        parts_check = action.split()
        service_check = parts_check[1] if len(parts_check) > 1 else ""
        previous_same_service_rollbacks = [
            a for a in self.action_history[:-1]
            if a.startswith(f"rollback_deploy {service_check}")
        ]
        if previous_same_service_rollbacks:
            return 0.0

        if action in correct_actions:
            self.state_data["services"][service] = "healthy"
            # Also clear downstream degradation caused by this service
            if service == "payment-service":
                self.state_data["services"]["web-frontend"] = "healthy"
            self.state_data["logs"].append(
                f"[SYSTEM] {service} rolled back to {version} successfully."
                " Memory usage stable."
            )
            return 0.5

        return 0.0

    # -------------------------
    # CHECK IF EPISODE IS DONE
    # -------------------------
    def _check_done(self) -> bool:
        win = self.current_task.get("win_condition", {})
        required_services = win.get("all_services", [])
        required_status = win.get("required_status", "healthy")
        required_action = win.get("required_action_taken", None)

        all_healthy = all(
            self.state_data["services"].get(s) == required_status
            for s in required_services
        )

        if required_action:
            return all_healthy and required_action in self.action_history

        return all_healthy

    # -------------------------
    # BUILD OBSERVATION
    # -------------------------
    def _build_observation(self) -> dict:
        return {
            "task_id": self.current_task["task_id"],
            "description": self.current_task["description"],
            "services": self.state_data["services"],
            "alerts": self.state_data["alerts"],
            "logs": self.state_data["logs"][-6:],
            "step": self.step_count,
            "max_steps": self.current_task["max_steps"]
        }
