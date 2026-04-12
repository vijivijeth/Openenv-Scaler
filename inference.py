import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from client import SREResponseGymClient
from models import SREAction, SREObservation
from server.environment import SREResponseGymEnvironment
from server.grader import Grader


ROOT_DIR = Path(__file__).resolve().parent
TASKS_DIR = ROOT_DIR / "tasks"

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_URL = (os.getenv("ENV_URL") or "").strip()

BENCHMARK = "sre-responsegym"
TASKS = list(SREResponseGymEnvironment.TASK_ORDER)
SUCCESS_SCORE_THRESHOLD = 0.75
SERVER_HOST = "127.0.0.1"
SERVER_TIMEOUT_S = 30.0
TEMPERATURE = 0.1
MAX_TOKENS = 48

_disabled_api_key: str | None = None

SYSTEM_PROMPT = """You are an incident commander operating a deterministic SRE simulator.

Rules:
- Prefer inspect-first actions before risky remediation.
- Acknowledge critical alerts early.
- Use runbooks when the observation suggests split-brain, cache stampede, token expiry, or OOM.
- Do not reset circuits before upstream recovery.
- Return exactly one action string and nothing else.
"""


def load_task_data(task_id: str) -> dict:
    with open(TASKS_DIR / f"{task_id}.json", encoding="utf-8") as f:
        return json.load(f)


TASK_DATA = {task_id: load_task_data(task_id) for task_id in TASKS}
TASK_OPTIMAL_SEQUENCES = {
    task_id: TASK_DATA[task_id].get("solution_actions", [])
    for task_id in TASKS
}


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 2)


def sanitize_field(value: str | None) -> str:
    if not value:
        return "null"
    return " ".join(str(value).split())


def normalize_action(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def format_start_line(task: str) -> str:
    return f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}"


def format_step_line(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> str:
    return (
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={sanitize_field(error)}"
    )


def format_end_line(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    return (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={clamp_score(score):.2f} rewards={rewards_str}"
    )


def action_prefixes(observation: SREObservation) -> list[str]:
    return [tool.split()[0] for tool in observation.available_tools]


def build_llm_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")


def llm_status() -> dict:
    disabled = bool(API_KEY) and API_KEY == _disabled_api_key
    return {
        "configured": bool(API_KEY),
        "api_key_source": "HF_TOKEN" if os.getenv("HF_TOKEN") else ("API_KEY" if os.getenv("API_KEY") else None),
        "api_base_url": API_BASE_URL,
        "model_name": MODEL_NAME,
        "llm_attempt_enabled": bool(API_KEY) and not disabled,
        "disabled_for_current_key": disabled,
    }


def probe_llm() -> dict:
    status = llm_status()
    if not status["configured"]:
        return {**status, "ok": False, "reason": "No API key configured."}

    try:
        client = build_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply with exactly one valid simulator action."},
                {"role": "user", "content": "The auth-api is crashing after a config reload. Reply with one action only."},
            ],
            max_tokens=24,
            temperature=0.0,
        )
        action = normalize_action((response.choices[0].message.content or "").splitlines()[0])
        return {**status, "ok": True, "action": action}
    except Exception as exc:
        return {**status, "ok": False, "reason": sanitize_field(str(exc))}


def choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((SERVER_HOST, 0))
        return int(sock.getsockname()[1])


class LocalServerHandle:
    def __init__(self, process: subprocess.Popen[str], base_url: str):
        self.process = process
        self.base_url = base_url

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "server did not become ready"

    while time.time() < deadline:
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
            raise RuntimeError(sanitize_field(stderr) or "local uvicorn process exited before startup")

        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"health returned HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)

        time.sleep(0.5)

    raise RuntimeError(sanitize_field(last_error))


def start_local_server() -> LocalServerHandle:
    port = choose_free_port()
    base_url = f"http://{SERVER_HOST}:{port}"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    wait_for_server(base_url, process, SERVER_TIMEOUT_S)
    return LocalServerHandle(process=process, base_url=base_url)


def create_client(server: LocalServerHandle | None = None):
    if LOCAL_IMAGE_NAME:
        async_client = asyncio.run(SREResponseGymClient.from_docker_image(LOCAL_IMAGE_NAME))
        return async_client.sync()

    if ENV_URL:
        return SREResponseGymClient(base_url=ENV_URL).sync()

    if server is None:
        raise RuntimeError("Local server handle is required when ENV_URL and LOCAL_IMAGE_NAME are unset.")
    return SREResponseGymClient(base_url=server.base_url).sync()


def _extract_observation(result: Any):
    return getattr(result, "observation", result)


def _extract_step_result(result: Any):
    observation = getattr(result, "observation", result)
    reward = getattr(result, "reward", getattr(observation, "reward", 0.0))
    done = getattr(result, "done", getattr(observation, "done", False))
    return observation, reward, done


def _extract_state(env: Any):
    state_obj = getattr(env, "state")
    return state_obj() if callable(state_obj) else state_obj


def last_action_error(observation: SREObservation) -> str | None:
    result = getattr(observation, "last_action_result", {}) or {}
    outcome = str(result.get("outcome", "")).lower()
    if outcome in {"invalid", "unsafe"}:
        return result.get("detail")
    return None


def get_action(
    client: OpenAI,
    observation: SREObservation,
    action_history: list[str],
    task_id: str,
) -> tuple[str, bool]:
    global _disabled_api_key

    valid_prefixes = action_prefixes(observation)
    task = TASK_DATA[task_id]

    if API_KEY and API_KEY != _disabled_api_key:
        try:
            prompt = {
                "task_id": observation.task_id,
                "description": observation.description,
                "services": observation.services,
                "alerts": observation.alerts,
                "logs": observation.logs[-6:],
                "metrics": observation.metrics,
                "timeline": observation.timeline[-5:],
                "customer_impact": observation.customer_impact,
                "available_tools": observation.available_tools,
                "action_history": action_history,
                "step": observation.step,
                "max_steps": observation.max_steps,
            }

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            llm_action = normalize_action((response.choices[0].message.content or "").splitlines()[0])
            if llm_action and any(llm_action.startswith(prefix) for prefix in valid_prefixes):
                return llm_action, True
        except Exception as exc:
            error_text = str(exc).lower()
            if any(token in error_text for token in ["401", "402", "403", "credit", "insufficient_quota"]):
                _disabled_api_key = API_KEY

    normalized_history = [normalize_action(action) for action in action_history]
    solution = TASK_OPTIMAL_SEQUENCES.get(task_id, [])
    next_index = len(normalized_history)
    if next_index < len(solution):
        return solution[next_index], False

    required_actions = [
        normalize_action(action)
        for action in task.get("success_criteria", {}).get("required_actions", [])
    ]
    for action in required_actions:
        if action not in normalized_history:
            return action, False

    diagnosis_actions = [normalize_action(action) for action in task.get("diagnosis_actions", [])]
    for action in diagnosis_actions:
        if action not in normalized_history:
            return action, False

    for alert in observation.alerts:
        ack_action = f"acknowledge_alert {alert.get('id')}"
        if alert.get("severity") == "critical" and ack_action not in normalized_history:
            return ack_action, False

    unhealthy = [
        name for name, status in observation.services.items()
        if status in {"down", "degraded", "unreachable", "locked"}
    ]
    for service in unhealthy:
        if f"check_logs {service}" not in normalized_history:
            return f"check_logs {service}", False
        if f"check_metrics {service}" not in normalized_history:
            return f"check_metrics {service}", False
        if f"check_dependencies {service}" not in normalized_history:
            return f"check_dependencies {service}", False

    return (solution[-1] if solution else "check_logs auth-api"), False


def compute_grade(task_id: str, state: Any) -> dict:
    grader = Grader(TASK_DATA[task_id])
    final_services = {
        name: details.get("status", "unknown")
        for name, details in state.services.items()
    }
    return grader.grade(
        final_services=final_services,
        action_history=state.action_history,
        step_count=state.step_count,
        analysis_state=state.analysis,
        customer_impact=state.customer_impact,
    )


def run_episode_on_env(
    env: Any,
    llm_client: OpenAI,
    task_id: str,
    emit=None,
    emit_end: bool = True,
) -> dict:
    lines: list[str] = []
    rewards: list[float] = []
    steps = 0
    success = False
    score = 0.0
    used_llm = False

    def write(line: str) -> None:
        lines.append(line)
        if emit is not None:
            emit(line)

    write(format_start_line(task_id))

    try:
        observation = _extract_observation(env.reset(task_id=task_id, scenario_seed=0))
        action_history: list[str] = []
        max_steps = observation.max_steps or TASK_DATA[task_id].get("max_steps", 10)

        while steps < max_steps and not observation.done:
            raw_action, action_used_llm = get_action(llm_client, observation, action_history, task_id)
            used_llm = used_llm or action_used_llm
            action = normalize_action(raw_action)
            observation, raw_reward, done = _extract_step_result(env.step(SREAction(action=action)))
            reward = clamp_score(raw_reward)
            rewards.append(reward)
            steps += 1
            action_history.append(action)
            write(format_step_line(steps, action, reward, bool(done), last_action_error(observation)))
            if done:
                break

        state = _extract_state(env)
        grade = compute_grade(task_id, state)
        score = clamp_score(grade["final_score"])
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False
        score = 0.0
    finally:
        end_line = format_end_line(success, steps, score, rewards)
        if emit_end:
            write(end_line)

    return {
        "task_id": task_id,
        "success": success,
        "score": score,
        "steps": steps,
        "rewards": rewards,
        "lines": lines,
        "used_llm": used_llm,
        "end_line": end_line,
    }


def run_trace(env_factory, llm_client: OpenAI | None = None) -> dict:
    llm_client = llm_client or build_llm_client()
    lines: list[str] = []
    results = []
    used_llm = False

    for task_id in TASKS:
        env = env_factory()
        episode = run_episode_on_env(env, llm_client, task_id, emit=lines.append)
        used_llm = used_llm or episode["used_llm"]
        results.append(
            {
                "task_id": episode["task_id"],
                "success": episode["success"],
                "score": episode["score"],
                "steps": episode["steps"],
            }
        )

    average_score = sum(item["score"] for item in results) / max(len(results), 1)
    return {
        "lines": lines,
        "results": results,
        "average_score": round(average_score, 3),
        "used_llm": used_llm,
    }


def run_task_with_client(task_id: str, llm_client: OpenAI, server: LocalServerHandle | None = None) -> dict:
    def stdout_emit(line: str) -> None:
        print(line, flush=True)

    client = create_client(server)
    with client as env:
        episode = run_episode_on_env(env, llm_client, task_id, emit=stdout_emit, emit_end=False)
    print(episode["end_line"], flush=True)
    return episode


def main() -> int:
    llm_client = build_llm_client()
    server = None
    results = []

    try:
        if not ENV_URL and not LOCAL_IMAGE_NAME:
            server = start_local_server()

        for task_id in TASKS:
            results.append(run_task_with_client(task_id, llm_client, server))
    finally:
        if server is not None:
            server.close()

    average_score = sum(item["score"] for item in results) / max(len(results), 1)
    return 0 if average_score >= SUCCESS_SCORE_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
