"""
Quick local verification before deploying.
Run: python smoke_test.py
"""

import json
import sys
from pathlib import Path

import requests

from client import SREResponseGymClient
from models import SREAction
from server.environment import SREResponseGymEnvironment


BASE = "http://localhost:7860"
ROOT = Path(__file__).resolve().parent
PASS = 0
FAIL = 0
TASKS = list(SREResponseGymEnvironment.TASK_ORDER)


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [OK] {name}")
        PASS += 1
    else:
        suffix = f" {detail}" if detail else ""
        print(f"  [FAIL] {name}{suffix}")
        FAIL += 1


def load_task(task_id: str) -> dict:
    with open(ROOT / "tasks" / f"{task_id}.json", encoding="utf-8") as f:
        return json.load(f)


print("\n SRE-ResponseGym V2 Smoke Test")
print("=" * 44)

print("\n[1] Health check")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    payload = r.json()
    check("GET /health returns 200", r.status_code == 200)
    check("Status is healthy", payload.get("status") == "healthy")
except Exception as e:
    check("GET /health reachable", False, str(e))

print("\n[2] Metadata")
try:
    r = requests.get(f"{BASE}/metadata", timeout=5)
    payload = r.json()
    check("GET /metadata returns 200", r.status_code == 200)
    check("Metadata has environment name", payload.get("name") == "SRE-ResponseGym")
    check("Metadata has version", bool(payload.get("version")))
except Exception as e:
    check("GET /metadata reachable", False, str(e))

print("\n[3] Environment info")
try:
    r = requests.get(f"{BASE}/environment-info", timeout=5)
    payload = r.json()
    check("GET /environment-info returns 200", r.status_code == 200)
    check("Reports 9 tasks", payload.get("total_tasks") == 9)
    check("Concurrent sessions enabled", payload.get("supports_concurrent_sessions") is True)
except Exception as e:
    check("GET /environment-info reachable", False, str(e))

print("\n[4] Task catalog")
try:
    r = requests.get(f"{BASE}/tasks", timeout=5)
    tasks = r.json().get("tasks", [])
    check("GET /tasks returns 200", r.status_code == 200)
    check("Catalog has 9 tasks", len(tasks) == 9)
    check("Catalog includes extreme task", any(task.get("task_id") == "task_extreme" for task in tasks))
except Exception as e:
    check("GET /tasks reachable", False, str(e))

print("\n[5] OpenEnv client flow")
try:
    with SREResponseGymClient(base_url=BASE).sync() as env:
        for task_id in TASKS:
            result = env.reset(task_id=task_id, scenario_seed=0)
            check(f"reset() task={task_id}", result.observation.task_id == task_id)
            check(f"reset() max_steps for {task_id}", result.observation.max_steps == load_task(task_id)["max_steps"])

        env.reset(task_id="task_easy", scenario_seed=0)
        result = env.step(SREAction(action="acknowledge_alert 101"))
        check("step() returns typed observation", result.observation.task_id == "task_easy")
        check("step() reward in valid range", 0.0 <= float(result.reward or 0.0) <= 1.0)
        state = env.state()
        check("state() includes analysis", isinstance(state.analysis, dict) and "redundant_actions" in state.analysis)
        check("state() includes metrics", bool(state.metrics))
except Exception as e:
    check("OpenEnv client flow works", False, str(e))

print("\n[6] HTTP grade/history")
try:
    grade = requests.get(f"{BASE}/grade", timeout=5)
    history = requests.get(f"{BASE}/history", timeout=5)
    grade_payload = grade.json()
    check("GET /grade returns 200", grade.status_code == 200)
    check("Grade has new rubric keys", set(grade_payload.get("breakdown", {}).keys()) == {
        "resolution",
        "diagnosis_accuracy",
        "operational_safety",
        "efficiency",
        "incident_hygiene",
    })
    check("GET /history returns 200", history.status_code == 200)
except Exception as e:
    check("GET /grade and /history reachable", False, str(e))

print("\n[6.5] LLM status")
try:
    status = requests.get(f"{BASE}/llm-status", timeout=5)
    probe = requests.post(f"{BASE}/llm-probe", timeout=20)
    status_payload = status.json()
    probe_payload = probe.json()
    check("GET /llm-status returns 200", status.status_code == 200)
    check("LLM status includes configured field", "configured" in status_payload)
    check("POST /llm-probe returns 200", probe.status_code == 200)
    check("LLM probe includes ok field", "ok" in probe_payload)
except Exception as e:
    check("LLM status/probe reachable", False, str(e))

print("\n[7] Dashboard session flow")
try:
    reset = requests.post(
        f"{BASE}/dashboard/session",
        json={"task_id": "task_hard", "scenario_seed": 0},
        timeout=5,
    )
    payload = reset.json()
    session_id = payload.get("session_id", "")
    check("POST /dashboard/session returns 200", reset.status_code == 200)
    check("Dashboard reset returns session_id", bool(session_id))
    step = requests.post(
        f"{BASE}/dashboard/session/{session_id}/action",
        json={"action": "check_deploy payment-service"},
        timeout=5,
    )
    state = requests.get(f"{BASE}/dashboard/session/{session_id}/state", timeout=5)
    grade = requests.get(f"{BASE}/dashboard/session/{session_id}/grade", timeout=5)
    check("Dashboard step returns 200", step.status_code == 200)
    check("Dashboard state returns 200", state.status_code == 200)
    check("Dashboard grade returns 200", grade.status_code == 200)
    requests.delete(f"{BASE}/dashboard/session/{session_id}", timeout=5)
except Exception as e:
    check("Dashboard session flow works", False, str(e))

print("\n[8] Benchmark")
try:
    r = requests.post(f"{BASE}/benchmark", timeout=30)
    payload = r.json()
    check("POST /benchmark returns 200", r.status_code == 200)
    check("Benchmark reports 9 tasks", payload.get("total_tasks") == 9)
    check("Benchmark average score is strong", float(payload.get("average_score", 0.0)) >= 0.85)
except Exception as e:
    check("POST /benchmark reachable", False, str(e))

print("\n[9] Inference trace")
try:
    r = requests.post(f"{BASE}/inference-trace", timeout=60)
    payload = r.json()
    lines = payload.get("lines", [])
    check("POST /inference-trace returns 200", r.status_code == 200)
    check("Inference trace returns lines", len(lines) > 0)
    check("Inference trace contains START lines", any(line.startswith("[START]") for line in lines))
    check("Inference trace contains END lines", any(line.startswith("[END]") for line in lines))
except Exception as e:
    check("POST /inference-trace reachable", False, str(e))

print("\n" + "=" * 44)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed")
if FAIL == 0:
    print("  All checks passed - ready to deploy!")
else:
    print(f"  {FAIL} check(s) failed - fix before deploying")
print("=" * 44 + "\n")

sys.exit(0 if FAIL == 0 else 1)
