class Grader:

    def __init__(self, task: dict):
        self.task = task

    def _clamp(self, value: float) -> float:
        return round(max(0.01, min(0.99, value)), 3)

    def grade(self, final_services: dict, action_history: list,
              step_count: int) -> dict:

        resolution = self._clamp(self._score_resolution(final_services))
        efficiency = self._clamp(self._score_efficiency(step_count))
        root_cause = self._clamp(self._score_root_cause(action_history))

        trap = self.task.get("trap", {})
        wrong_action = trap.get("wrong_action", trap.get("wrong_first_action", ""))
        trap_hits = sum(1 for a in action_history if a == wrong_action)

        solution = self.task.get("solution_actions", [])
        actions_in_solution = sum(1 for a in action_history if a in solution)
        solution_coverage = self._clamp(
            actions_in_solution / len(solution) if solution else 0.5
        )

        raw = (
            0.5 * resolution +
            0.3 * efficiency +
            0.2 * root_cause
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
                "efficiency": efficiency,
                "root_cause_accuracy": root_cause
            },
            "analysis": {
                "trap_hits": trap_hits,
                "solution_coverage": solution_coverage,
                "optimal_steps": len(solution),
                "actual_steps": step_count,
                "unnecessary_actions": max(0, step_count - len(solution))
            },
            "actions_taken": action_history,
            "steps_used": step_count,
            "max_steps": self.task["max_steps"]
        }

    def _score_resolution(self, final_services: dict) -> float:
        win = self.task.get("win_condition", {})
        required = win.get("all_services", [])
        required_status = win.get("required_status", "healthy")

        if not required:
            return 0.5

        healthy_count = sum(
            1 for s in required
            if final_services.get(s) == required_status
        )
        return healthy_count / len(required)

    def _score_efficiency(self, step_count: int) -> float:
        max_steps = self.task.get("max_steps", 10)
        min_steps = len(self.task.get("solution_actions", [1]))

        if step_count <= min_steps:
            return 0.99

        if step_count >= max_steps:
            return 0.1

        range_steps = max_steps - min_steps
        steps_over = step_count - min_steps
        return round(1.0 - (steps_over / range_steps) * 0.9, 3)

    def _score_root_cause(self, action_history: list) -> float:
        correct_root = self.task.get("correct_root_cause", "")
        trap = self.task.get("trap", {})

        if not action_history:
            return 0.1

        wrong_first = trap.get("wrong_first_action", "")
        wrong_action = trap.get("wrong_action", "")
        first_action = action_history[0]

        if first_action in [wrong_first, wrong_action]:
            return 0.1

        for action in action_history[:2]:
            if correct_root in action:
                return 0.99

        for action in action_history:
            if correct_root in action:
                return 0.5

        required_action = self.task.get(
            "win_condition", {}
        ).get("required_action_taken", "")
        if required_action and required_action in action_history:
            return 0.8

        return 0.1