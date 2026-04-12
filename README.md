# OpenEnv SRE-ResponseGym V2

SRE-ResponseGym V2 is a deterministic OpenEnv environment for evaluating agents
on realistic SRE incident command workflows. Agents must investigate partial
evidence, avoid unsafe actions, communicate status updates, and recover service
health under time pressure.

## Why This Submission Is Strong

This project is intentionally designed to score well across the five hackathon
categories.

- Real-world utility: incidents model production failure modes such as bad
  deploys, connection leaks, split-brain failovers, and circuit breaker
  recovery.
- Task and grader quality: success depends on diagnosis, safety, efficiency, and
  incident hygiene instead of a single end-state check.
- Environment design: every action advances incident time, affects customer
  impact, and can worsen or irreversibly damage the system.
- Code quality and spec compliance: typed OpenEnv models, deterministic task
  data, validation coverage, and concurrent-session-friendly dashboard helpers.
- Creativity and novelty: the benchmark evaluates incident-command behavior, not
  just tool selection or single-step remediation.

## Canonical Incident Families

### Service-Level Incidents

| Task | Difficulty | Scenario |
|------|------------|----------|
| `task_easy` | Easy | Single auth-api crash loop after config reload |
| `task_noisy_alert` | Easy | False cache lead hiding a checkout-api crash |
| `task_token_expiry` | Medium | Expired signing bundle in token-broker |

### Platform Incidents

| Task | Difficulty | Scenario |
|------|------------|----------|
| `task_medium` | Medium | Database connection storm cascading into auth |
| `task_hard` | Hard | Bad payment-service deploy with memory leak |
| `task_pool_exhaustion` | Hard | Connection leak in ledger-api |

### Distributed Incidents

| Task | Difficulty | Scenario |
|------|------------|----------|
| `task_expert` | Expert | Dependency restore ordering across cache and gateway |
| `task_trap` | Expert | Split-brain failover with irreversible restart trap |
| `task_extreme` | Extreme | Multi-cause SEV1 with deploy, cache, and breaker failures |

## Action Grammar

The public action space stays text-based for OpenEnv compatibility:

- `check_logs <service>`
- `check_metrics <service>`
- `check_deploy <service>`
- `check_dependencies <service>`
- `query_runbook <keyword>`
- `restart_service <service>`
- `rollback_deploy <service> <version>`
- `scale_service <service> <replicas>`
- `drain_traffic <service>`
- `failover_db <cluster>`
- `isolate_az <zone>`
- `reset_circuit_breaker <service>`
- `acknowledge_alert <id>`
- `post_status <severity> <message>`

## Observation and State

Each `SREObservation` includes:

- `task_id`
- `description`
- `services`
- `alerts`
- `logs`
- `metrics`
- `timeline`
- `customer_impact`
- `available_tools`
- `last_action_result`
- `step`
- `max_steps`
- `reward`
- `done`

The typed `SREState` also exposes deployment and grading context such as:

- `deploy_history`
- `dependency_graph`
- `runbooks`
- `action_history`
- `analysis`
- `hidden_causes`
- `scenario_seed`
- `minutes_elapsed`

## Grading Rubric

The final score is deterministic and clamped to `[0.01, 0.99]`.

- `resolution`: 35%
- `diagnosis_accuracy`: 20%
- `operational_safety`: 20%
- `efficiency`: 15%
- `incident_hygiene`: 10%

The grader also reports:

- forbidden action hits
- irreversible mistakes
- redundant actions
- time to first diagnosis
- peak users affected percentage

## API Surface

OpenEnv base endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`

Environment helpers:

- `GET /tasks`
- `GET /task/{task_id}`
- `GET /environment-info`
- `GET /grade`
- `GET /history`
- `POST /benchmark`
- `POST /inference-trace`
- `GET /llm-status`
- `POST /llm-probe`

Dashboard session endpoints:

- `POST /dashboard/session`
- `POST /dashboard/session/{session_id}/action`
- `GET /dashboard/session/{session_id}/state`
- `GET /dashboard/session/{session_id}/grade`
- `DELETE /dashboard/session/{session_id}`

## Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Validation:

```bash
python smoke_test.py
openenv validate
```

## LLM Configuration

The inference runner can use any of these environment variables for the router
token:

- `HF_TOKEN`
- `API_KEY`
- `HF_API_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`
- `OPENAI_API_KEY`

You can also create a local `.env` file by copying `.env.example`.

Example:

```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

Quick verification:

```bash
python llm_smoke_test.py
```

Or through the app:

- `GET /llm-status`
- `POST /llm-probe`

## Inference Runner

`inference.py` stays at the repo root and preserves the required stdout format:

- `[START]`
- `[STEP]`
- `[END]`

Execution modes:

- `ENV_URL=<running_env>` connects to a deployed environment.
- `LOCAL_IMAGE_NAME=<docker_image>` uses the OpenEnv Docker client flow.
- With neither set, the script starts the local FastAPI server automatically.

The policy is inspect-first, LLM-assisted when credentials are available, and
deterministic when they are not. That keeps the benchmark reproducible while
still supporting real model-driven inference.

## Hugging Face / Deployment Checklist

Before submission:

1. Push the latest repo state to GitHub.
2. Redeploy the Hugging Face Space with the updated Docker image.
3. Add `HF_TOKEN` as a Space secret if you want true LLM-backed inference.
4. Confirm these routes on the live deployment:
   - `/tasks`
   - `/environment-info`
   - `/benchmark`
   - `/inference-trace`
   - `/llm-status`
5. Run the local smoke test and `openenv validate`.

## Architecture Note

This environment is useful for agent evaluation because it combines:

- partial observability
- explicit tool use
- cascading service dependencies
- time-sensitive customer impact
- irreversible operational mistakes

That makes it much closer to real production incident response than toy
single-step tool selection tasks while still remaining reproducible and
benchmarkable.
