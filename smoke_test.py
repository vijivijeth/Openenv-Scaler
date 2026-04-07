"""
smoke_test.py — Quick local verification before deploying
Run: python smoke_test.py
"""
import requests
import sys

BASE = "http://localhost:7860"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name} {detail}")
        FAIL += 1


print("\n SRE-ResponseGym Smoke Test")
print("=" * 40)

# Health
print("\n[1] Health check")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    check("GET /health returns 200", r.status_code == 200)
    check("Status is healthy", r.json().get("status") == "healthy")
except Exception as e:
    check("GET /health reachable", False, str(e))

# Metadata
print("\n[2] Metadata")
try:
    r = requests.get(f"{BASE}/metadata", timeout=5)
    check("GET /metadata returns 200", r.status_code == 200)
    check("Has action_space", "action_space" in r.json())
except Exception as e:
    check("GET /metadata reachable", False, str(e))

# Tasks
print("\n[3] Task listing")
try:
    r = requests.get(f"{BASE}/tasks", timeout=5)
    check("GET /tasks returns 200", r.status_code == 200)
    tasks = r.json().get("tasks", [])
    check("Has 6+ tasks", len(tasks) >= 6)
except Exception as e:
    check("GET /tasks reachable", False, str(e))

# Reset
print("\n[4] Reset")
TASKS = [
    "task_easy", "task_medium", "task_hard",
    "task_expert", "task_trap", "task_extreme"
]
for task_id in TASKS:
    try:
        r = requests.post(
            f"{BASE}/reset",
            json={"task_id": task_id},
            timeout=5
        )
        check(
            f"POST /reset task={task_id}",
            r.status_code == 200 and "observation" in r.json()
        )
    except Exception as e:
        check(f"POST /reset task={task_id}", False, str(e))

# Step
print("\n[5] Step")
try:
    requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
    r = requests.post(
        f"{BASE}/step",
        json={"action": "check_logs auth-api"},
        timeout=5
    )
    data = r.json()
    check("POST /step returns 200", r.status_code == 200)
    check("Has observation", "observation" in data)
    check("Has reward", "reward" in data)
    check("Has done", "done" in data)
    check(
        "Reward in valid range",
        -1.0 <= data.get("reward", 0) <= 1.0
    )
except Exception as e:
    check("POST /step works", False, str(e))

# State
print("\n[6] State")
try:
    r = requests.get(f"{BASE}/state", timeout=5)
    check("GET /state returns 200", r.status_code == 200)
    check("Has services", "services" in r.json())
except Exception as e:
    check("GET /state reachable", False, str(e))

# Grade
print("\n[7] Grade")
try:
    r = requests.get(f"{BASE}/grade", timeout=5)
    check("GET /grade returns 200", r.status_code == 200)
    score = r.json().get("final_score", -1)
    check("Score between 0 and 1", 0.0 <= score <= 1.0)
    check("Has breakdown", "breakdown" in r.json())
    check("Has letter_grade", "letter_grade" in r.json())
except Exception as e:
    check("GET /grade reachable", False, str(e))

# History
print("\n[8] History")
try:
    r = requests.get(f"{BASE}/history", timeout=5)
    check("GET /history returns 200", r.status_code == 200)
except Exception as e:
    check("GET /history reachable", False, str(e))

# Summary
print("\n" + "=" * 40)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed")
if FAIL == 0:
    print("  All checks passed — ready to deploy!")
else:
    print(f"  {FAIL} check(s) failed — fix before deploying")
print("=" * 40 + "\n")

sys.exit(0 if FAIL == 0 else 1)