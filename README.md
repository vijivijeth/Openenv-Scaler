# SRE-ResponseGym

**AI Incident Response Benchmark — OpenEnv Compatible**

> Train and evaluate AI agents on realistic Site Reliability Engineering 
> incident response scenarios. From simple crashes to full platform 
> meltdowns — agents must read logs, trace root causes, and remediate 
> production failures without causing further damage.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![Tasks](https://img.shields.io/badge/Tasks-6-orange)]()
[![Docker](https://img.shields.io/badge/Docker-ready-green)]()

---

## Why This Environment

Real SRE engineers face exactly these decisions daily. The environment models:

- **Cascading failures** — fixing the wrong service first makes things worse
- **Bad deployments** — restart will never fix a memory leak, rollback required
- **Network partitions** — restarting during split-brain causes data corruption
- **Circuit breakers** — must be manually reset after upstream recovery
- **Dependency ordering** — services must come back up in the right sequence

These mechanics create meaningful reward signals that genuinely differentiate 
intelligent agents from random action agents.

---

## Tasks

| Task | Difficulty | Scenario | Key Mechanic |
|------|-----------|----------|--------------|
| task_easy | Easy | Single service crash | Basic restart |
| task_medium | Medium | DB cascade failure | Trap: wrong first action penalized |
| task_hard | Hard | Bad deployment | Rollback required, restart always fails |
| task_expert | Expert | Dependency chain | Correct restoration order required |
| task_trap | Expert | Network partition | Restart causes data corruption |
| task_extreme | Extreme | Platform meltdown | Three root causes, circuit breaker |

---

## Action Space

| Action | Example | Effect |
|--------|---------|--------|
| `restart_service <name>` | `restart_service auth-api` | Restarts service if healthy root cause resolved |
| `check_logs <name>` | `check_logs database` | Appends log context, +0.05 reward |
| `query_runbook <keyword>` | `query_runbook OOMKilled` | Returns remediation procedure, +0.10 reward |
| `rollback_deploy <name> <ver>` | `rollback_deploy payment-service v3.2.0` | Rolls back bad deployment, +0.50 reward |
| `isolate_az <zone>` | `isolate_az us-east-1a` | Fences affected availability zone, +0.30 reward |
| `failover_db <name>` | `failover_db database-primary` | Promotes replica to primary, +0.40 reward |
| `reset_circuit_breaker <name>` | `reset_circuit_breaker edge-gateway` | Resets tripped circuit breaker, +0.50 reward |
| `acknowledge_alert <id>` | `acknowledge_alert 1` | Clears alert, +0.05 reward |

---

## Reward Function
Final Score = 0.5 × Resolution + 0.3 × Efficiency + 0.2 × Root Cause Accuracy
| Component | Weight | Description |
|-----------|--------|-------------|
| Resolution | 50% | Fraction of required services restored to healthy |
| Efficiency | 30% | Steps used vs optimal steps — penalizes waste |
| Root Cause Accuracy | 20% | Did agent identify correct root cause first |

**Trap mechanics:** Wrong actions apply -0.3 to -0.5 penalty and leave 
services in a worse state, forcing the agent to recover. Restarting during 
a network partition costs -0.5.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/metadata` | GET | Environment metadata and capabilities |
| `/tasks` | GET | List all 6 tasks with descriptions |
| `/reset` | POST | Start episode: `{"task_id": "task_easy"}` |
| `/step` | POST | Take action: `{"action": "restart_service auth-api"}` |
| `/state` | GET | Current environment state |
| `/grade` | GET | Full score breakdown with letter grade |
| `/history` | GET | Episode history and score trends |
| `/benchmark` | POST | Run all tasks and return full benchmark report |

### Quick Start with curl
```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_hard"}'

# Send an action  
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "check_logs payment-service"}'

# Get score
curl http://localhost:7860/grade

# Run full benchmark
curl -X POST http://localhost:7860/benchmark
```

---

## Setup
```bash
# Local
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t sre-responsegym .
docker run -p 7860:7860 sre-responsegym

# Validate
python smoke_test.py
openenv validate
```

---

## Inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token_here

python inference.py
```

---

## Baseline Scores

| Task | Score | Grade | Steps |
|------|-------|-------|-------|
| task_easy | 0.932 | A | 2 |
| task_medium | 0.875 | A | 6 |
| task_hard | 0.960 | A | 2 |
| task_expert | 1.000 | A | 5 |
| task_trap | 0.793 | B | 5 |
| task_extreme | 0.793 | B | 6 |
| **Average** | **0.892** | **A** | — |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (default: HF Router) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API key |
| `ENV_URL` | Environment URL (default: localhost:7860) |