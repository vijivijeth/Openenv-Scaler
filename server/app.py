import sys
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import SREAction, SREObservation
from server.environment import SREResponseGymEnvironment
from inference import llm_status, probe_llm, run_trace


app = create_fastapi_app(SREResponseGymEnvironment, SREAction, SREObservation)
app.mount("/static", StaticFiles(directory="static"), name="static")


DASHBOARD_SESSIONS: dict[str, SREResponseGymEnvironment] = {}
DASHBOARD_LOCK = Lock()


class DashboardResetRequest(BaseModel):
    task_id: str = "task_easy"
    scenario_seed: int | None = None


class DashboardActionRequest(BaseModel):
    action: str


def _neutral_grade() -> dict:
    return {
        "final_score": 0.5,
        "letter_grade": "C",
        "breakdown": {
            "resolution": 0.5,
            "diagnosis_accuracy": 0.5,
            "operational_safety": 0.5,
            "efficiency": 0.5,
            "incident_hygiene": 0.5,
        },
        "analysis": {},
        "actions_taken": [],
        "steps_used": 0,
        "max_steps": 0,
    }


def _task_summary(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "title": task.get("title", task["task_id"]),
        "difficulty": task.get("difficulty", "unknown"),
        "family": task.get("family", "unknown"),
        "description": task.get("description", ""),
        "max_steps": task.get("max_steps", 0),
        "solution_length": len(task.get("solution_actions", [])),
    }


def _require_task(task_id: str) -> dict:
    try:
        return SREResponseGymEnvironment.load_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'.") from exc


def _get_dashboard_env(session_id: str) -> SREResponseGymEnvironment:
    with DASHBOARD_LOCK:
        env = DASHBOARD_SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Dashboard session not found.")
    return env


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/tasks")
def list_tasks():
    return {"tasks": [_task_summary(task) for task in SREResponseGymEnvironment.task_catalog()]}


@app.get("/environment-info")
def environment_info():
    env = SREResponseGymEnvironment()
    metadata = env.get_metadata()
    return {
        "name": metadata.name,
        "version": metadata.version,
        "description": metadata.description,
        "domain": "Site Reliability Engineering",
        "supports_concurrent_sessions": SREResponseGymEnvironment.SUPPORTS_CONCURRENT_SESSIONS,
        "total_tasks": len(SREResponseGymEnvironment.TASK_ORDER),
        "available_tools": list(SREResponseGymEnvironment.AVAILABLE_TOOLS),
        "grading_axes": {
            "resolution": 0.35,
            "diagnosis_accuracy": 0.20,
            "operational_safety": 0.20,
            "efficiency": 0.15,
            "incident_hygiene": 0.10,
        },
        "families": ["service-level", "platform", "distributed"],
    }


@app.get("/task/{task_id}")
def get_task(task_id: str):
    task = _require_task(task_id)
    return {
        "task": _task_summary(task),
        "incident_commander_brief": task.get("incident_commander_brief", ""),
        "diagnosis_actions": task.get("diagnosis_actions", []),
        "hygiene_expectations": task.get("hygiene_expectations", {}),
    }


@app.get("/grade")
def grade():
    env = SREResponseGymEnvironment.snapshot_env()
    if env is None or env.current_task is None:
        return _neutral_grade()
    return env.get_grade()


@app.get("/history")
def get_history():
    return SREResponseGymEnvironment.history_summary()


@app.post("/benchmark")
def run_benchmark():
    results = []
    total_score = 0.0

    for task in SREResponseGymEnvironment.task_catalog():
        task_id = task["task_id"]
        env = SREResponseGymEnvironment()
        env.reset(task_id=task_id, scenario_seed=0)

        for action_str in task.get("solution_actions", []):
            obs = env.step(SREAction(action=action_str))
            if obs.done:
                break

        grade_payload = env.get_grade()
        total_score += grade_payload["final_score"]
        results.append(
            {
                "task_id": task_id,
                "title": task.get("title", task_id),
                "difficulty": task.get("difficulty", "unknown"),
                "family": task.get("family", "unknown"),
                "score": grade_payload["final_score"],
                "letter_grade": grade_payload["letter_grade"],
                "breakdown": grade_payload["breakdown"],
                "steps_used": env.step_count,
                "max_steps": task.get("max_steps", 0),
            }
        )

    task_count = max(len(results), 1)
    average_score = round(total_score / task_count, 3)
    overall_grade = (
        "A" if average_score >= 0.90 else
        "B" if average_score >= 0.75 else
        "C" if average_score >= 0.50 else "F"
    )
    return {
        "benchmark": "SRE-ResponseGym",
        "version": "2.0.0",
        "total_tasks": task_count,
        "average_score": average_score,
        "overall_grade": overall_grade,
        "results": results,
    }


@app.post("/inference-trace")
def inference_trace():
    trace = run_trace(SREResponseGymEnvironment)
    return {
        "benchmark": "sre-responsegym",
        "version": "2.0.0",
        "task_count": len(trace["results"]),
        "average_score": trace["average_score"],
        "used_llm": trace["used_llm"],
        "lines": trace["lines"],
        "results": trace["results"],
    }


@app.get("/llm-status")
def get_llm_status():
    return llm_status()


@app.post("/llm-probe")
def post_llm_probe():
    return probe_llm()


@app.post("/dashboard/session")
def dashboard_reset(request: DashboardResetRequest):
    _require_task(request.task_id)
    env = SREResponseGymEnvironment()
    observation = env.reset(task_id=request.task_id, scenario_seed=request.scenario_seed)
    session_id = str(uuid4())
    with DASHBOARD_LOCK:
        DASHBOARD_SESSIONS[session_id] = env
    return {"session_id": session_id, "observation": observation.model_dump()}


@app.post("/dashboard/session/{session_id}/action")
def dashboard_step(session_id: str, request: DashboardActionRequest):
    env = _get_dashboard_env(session_id)
    observation = env.step(SREAction(action=request.action))
    payload = {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
    }
    if observation.done:
        payload["grade"] = env.get_grade()
    return payload


@app.get("/dashboard/session/{session_id}/state")
def dashboard_state(session_id: str):
    env = _get_dashboard_env(session_id)
    return env.state.model_dump()


@app.get("/dashboard/session/{session_id}/grade")
def dashboard_grade(session_id: str):
    env = _get_dashboard_env(session_id)
    if env.current_task is None:
        return _neutral_grade()
    return env.get_grade()


@app.delete("/dashboard/session/{session_id}")
def dashboard_close(session_id: str):
    with DASHBOARD_LOCK:
        existed = DASHBOARD_SESSIONS.pop(session_id, None)
    return {"closed": existed is not None, "session_id": session_id}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
