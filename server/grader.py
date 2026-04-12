class Grader:
    """Deterministic multi-axis grader for incident-command scenarios."""

    def __init__(self, task: dict):
        self.task = task

    def _clamp(self, value: float) -> float:
        return round(max(0.01, min(0.99, value)), 3)

    def grade(
        self,
        final_services: dict,
        action_history: list,
        step_count: int,
        analysis_state: dict | None = None,
        customer_impact: dict | None = None,
    ) -> dict:
        analysis_state = analysis_state or {}
        customer_impact = customer_impact or {}

        resolution = self._clamp(self._score_resolution(final_services, action_history))
        diagnosis = self._clamp(self._score_diagnosis(action_history, analysis_state))
        safety = self._clamp(self._score_safety(analysis_state))
        efficiency = self._clamp(self._score_efficiency(step_count))
        hygiene = self._clamp(self._score_hygiene(action_history, analysis_state))

        raw = (
            0.35 * resolution
            + 0.20 * diagnosis
            + 0.20 * safety
            + 0.15 * efficiency
            + 0.10 * hygiene
        )
        final_score = self._clamp(raw)

        if final_score >= 0.9:
            letter = "A"
        elif final_score >= 0.75:
            letter = "B"
        elif final_score >= 0.5:
            letter = "C"
        else:
            letter = "F"

        return {
            "final_score": final_score,
            "letter_grade": letter,
            "breakdown": {
                "resolution": resolution,
                "diagnosis_accuracy": diagnosis,
                "operational_safety": safety,
                "efficiency": efficiency,
                "incident_hygiene": hygiene,
            },
            "analysis": {
                "forbidden_action_hits": analysis_state.get("forbidden_action_hits", 0),
                "irreversible_mistakes": analysis_state.get("irreversible_mistakes", 0),
                "redundant_actions": analysis_state.get("redundant_actions", 0),
                "time_to_first_diagnosis": analysis_state.get("time_to_first_diagnosis"),
                "peak_users_affected_pct": analysis_state.get("peak_users_affected_pct", 0.0),
                "final_customer_severity": customer_impact.get("severity", "sev4"),
                "final_users_affected_pct": customer_impact.get("users_affected_pct", 0.0),
            },
            "actions_taken": action_history,
            "steps_used": step_count,
            "max_steps": self.task.get("max_steps", 0),
        }

    def _score_resolution(self, final_services: dict, action_history: list) -> float:
        success = self.task.get("success_criteria", {})
        required_services = success.get("required_services", [])
        required_status = success.get("required_status", "healthy")
        required_actions = success.get("required_actions", [])

        if not required_services:
            return 0.5

        healthy_count = sum(
            1 for service in required_services if final_services.get(service) == required_status
        )
        score = healthy_count / len(required_services)

        if required_actions:
            completed_actions = sum(1 for action in required_actions if action in action_history)
            score *= 0.7 + 0.3 * (completed_actions / len(required_actions))

        return score

    def _score_diagnosis(self, action_history: list, analysis_state: dict) -> float:
        diagnosis_actions = [action.lower() for action in self.task.get("diagnosis_actions", [])]
        normalized_history = [action.lower() for action in action_history]

        if not normalized_history:
            return 0.1

        for action in normalized_history[:3]:
            if action in diagnosis_actions:
                return 0.99

        if analysis_state.get("time_to_first_diagnosis") is not None:
            return 0.65

        if any(action in diagnosis_actions for action in normalized_history):
            return 0.5

        return 0.1

    def _score_safety(self, analysis_state: dict) -> float:
        score = 0.99
        score -= 0.18 * analysis_state.get("forbidden_action_hits", 0)
        score -= 0.22 * analysis_state.get("irreversible_mistakes", 0)
        return score

    def _score_efficiency(self, step_count: int) -> float:
        max_steps = self.task.get("max_steps", 10)
        min_steps = max(1, len(self.task.get("solution_actions", [1])))

        if step_count <= min_steps:
            return 0.99
        if step_count >= max_steps:
            return 0.1

        range_steps = max(max_steps - min_steps, 1)
        steps_over = step_count - min_steps
        return 1.0 - (steps_over / range_steps) * 0.9

    def _score_hygiene(self, action_history: list, analysis_state: dict) -> float:
        normalized_history = [action.lower() for action in action_history]
        expectations = self.task.get("hygiene_expectations", {})

        score = 0.99

        if expectations.get("require_ack_critical"):
            critical_alert_count = len(
                [
                    alert
                    for alert in self.task.get("initial_state", {}).get("alerts", [])
                    if alert.get("severity") == "critical"
                ]
            )
            acknowledged = len(analysis_state.get("acknowledged_alert_ids", []))
            if critical_alert_count > 0 and acknowledged == 0:
                score -= 0.35
            elif critical_alert_count > 0 and acknowledged < min(1, critical_alert_count):
                score -= 0.15

        min_status_updates = expectations.get("min_status_updates", 0)
        status_updates = len(analysis_state.get("status_updates", []))
        if status_updates < min_status_updates:
            score -= 0.25

        score -= min(0.35, analysis_state.get("redundant_actions", 0) * 0.04)

        if any(action.startswith("post_status ") for action in normalized_history) and score < 0.99:
            score += 0.05

        return score
