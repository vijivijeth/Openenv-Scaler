from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.environment import IncidentEnv
from server.grader import Grader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SRE-ResponseGym", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
env = IncidentEnv()
current_task_data = {}

# -------------------------
# OPENENV TYPED MODELS
# -------------------------
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

class Reward(BaseModel):
    reward: float
    done: bool
    info: dict

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"

class StepRequest(BaseModel):
    action: str

# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    global current_task_data
    observation = env.reset(request.task_id)

    with open(f"tasks/{request.task_id}.json") as f:
        current_task_data = json.load(f)

    return {
        "observation": observation,
        "message": f"Episode started for {request.task_id}"
    }

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    if env.current_task is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first."
        )
    result = env.step(request.action)
    return result

@app.get("/state")
def state():
    if env.current_task is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first."
        )
    return env.state()

@app.get("/grade")
def grade():
    if env.current_task is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first."
        )

    grader = Grader(current_task_data)
    result = grader.grade(
        final_services=env.state_data["services"],
        action_history=env.action_history,
        step_count=env.step_count
    )
    return result

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "task_easy",
                "difficulty": "easy",
                "description": "Single service crash. Identify and restart."
            },
            {
                "task_id": "task_medium",
                "difficulty": "medium",
                "description": "Database cascade failure. Trace root cause."
            },
            {
                "task_id": "task_hard",
                "difficulty": "hard",
                "description": "Bad deployment with memory leak. Rollback required."
            },
            {
                "task_id": "task_expert",
                "difficulty": "expert",
                "description": "Multi-service cascade. Fix in correct dependency order."
            },
            {
                "task_id": "task_trap",
                "difficulty": "expert",
                "description": "Network partition. Restarting causes data corruption."
            },
            {
                "task_id": "task_extreme",
                "difficulty": "extreme",
                "description": "Full platform meltdown. Three independent root causes."
            }
        ]
    }
    
# -------------------------
# BENCHMARK RUNNER
# -------------------------
@app.post("/benchmark")
def run_benchmark():
    tasks = [
        "task_easy", "task_medium", "task_hard",
        "task_expert", "task_trap", "task_extreme"
    ]

    OPTIMAL_SEQUENCES = {
        "task_easy": [
            "check_logs auth-api",
            "restart_service auth-api"
        ],
        "task_medium": [
            "check_logs database",
            "restart_service database",
            "check_logs auth-api",
            "restart_service auth-api",
            "check_logs web-frontend",
            "restart_service web-frontend"
        ],
        "task_hard": [
            "check_logs payment-service",
            "rollback_deploy payment-service v3.2.0"
        ],
        "task_expert": [
            "check_logs user-service",
            "restart_service user-service",
            "check_logs order-service",
            "restart_service order-service",
            "restart_service edge-gateway"
        ],
        "task_trap": [
            "query_runbook split-brain",
            "isolate_az us-east-1a",
            "failover_db database-primary",
            "restart_service user-service",
            "restart_service payment-gateway"
        ],
        "task_extreme": [
            "check_logs user-service",
            "query_runbook OOMKilled",
            "rollback_deploy user-service v4.0.8",
            "check_logs payment-service",
            "rollback_deploy payment-service v2.3.0",
            "reset_circuit_breaker edge-gateway"
        ]
    }

    results = []
    total_score = 0.0

    for task_id in tasks:
        try:
            task_file = f"tasks/{task_id}.json"
            with open(task_file) as f:
                task_data = json.load(f)

            # Reset the environment
            env.reset(task_id)

            # Play through optimal sequence
            actions = OPTIMAL_SEQUENCES.get(task_id, [])
            for action in actions:
                result = env.step(action)
                if result.get("done"):
                    break

            # Grade the completed episode
            grader = Grader(task_data)
            grade = grader.grade(
                final_services=env.state_data["services"],
                action_history=env.action_history,
                step_count=env.step_count
            )

            results.append({
                "task_id": task_id,
                "difficulty": task_data.get("difficulty", "unknown"),
                "score": grade["final_score"],
                "letter_grade": grade["letter_grade"],
                "breakdown": grade["breakdown"],
                "analysis": grade["analysis"],
                "steps_used": env.step_count
            })
            total_score += grade["final_score"]

        except Exception as e:
            results.append({
                "task_id": task_id,
                "difficulty": "unknown",
                "score": 0.0,
                "letter_grade": "F",
                "error": str(e)
            })

    avg = round(total_score / len(tasks), 3)

    return {
        "benchmark": "SRE-ResponseGym",
        "version": "1.0.0",
        "total_tasks": len(tasks),
        "average_score": avg,
        "overall_grade": (
            "A" if avg >= 0.9 else
            "B" if avg >= 0.75 else
            "C" if avg >= 0.5 else "F"
        ),
        "results": results
    }

# -------------------------
# EPISODE HISTORY
# -------------------------
@app.get("/history")
def get_history():
    if env.current_task is None:
        return {"episodes": [], "total": 0}

    by_task = {}
    for ep in env.episode_history:
        tid = ep["task_id"]
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append({
            "episode_id": ep["episode_id"],
            "score": ep["score"],
            "steps": ep["steps"],
            "breakdown": ep["breakdown"]
        })

    return {
        "total_episodes": len(env.episode_history),
        "by_task": by_task,
        "recent": env.episode_history[-10:]
    }


# -------------------------
# SMOKE TEST ENDPOINT
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "SRE-ResponseGym",
        "version": "1.0.0",
        "tasks_available": 6,
        "endpoints": [
            "/reset", "/step", "/state", "/grade",
            "/tasks", "/benchmark", "/history", "/health"
        ]
    }


# -------------------------
# ENVIRONMENT METADATA
# -------------------------
@app.get("/metadata")
def metadata():
    return {
        "name": "SRE-ResponseGym",
        "version": "1.0.0",
        "description": "AI Incident Response Benchmark Environment",
        "domain": "Site Reliability Engineering",
        "total_tasks": 6,
        "difficulty_levels": ["easy", "medium", "hard", "expert", "extreme"],
        "action_space": [
            "restart_service <name>",
            "check_logs <name>",
            "query_runbook <keyword>",
            "rollback_deploy <name> <version>",
            "acknowledge_alert <id>",
            "isolate_az <zone>",
            "failover_db <name>",
            "reset_circuit_breaker <name>"
        ],
        "reward_range": [-1.0, 1.0],
        "grading_axes": {
            "resolution": "50% — were all required services restored",
            "efficiency": "30% — how quickly was it solved",
            "root_cause_accuracy": "20% — was the correct root cause identified first"
        },
        "special_mechanics": [
            "Trap actions that penalize obvious-but-wrong decisions",
            "Rollback mechanic for bad deployments",
            "Network partition isolation and database failover",
            "Circuit breaker reset for cascading failures",
            "Dependency ordering requirements"
        ]
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
