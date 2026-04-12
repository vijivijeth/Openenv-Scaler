"""
Quick check for the LLM path used by inference.py.

Run:
    python llm_smoke_test.py
"""

from inference import llm_status, probe_llm


def main() -> int:
    status = llm_status()
    print("LLM status")
    print(f"  configured: {status['configured']}")
    print(f"  api_key_source: {status['api_key_source']}")
    print(f"  api_base_url: {status['api_base_url']}")
    print(f"  model_name: {status['model_name']}")
    print(f"  llm_attempt_enabled: {status['llm_attempt_enabled']}")

    result = probe_llm()
    print("Probe result")
    print(f"  ok: {result['ok']}")
    if result.get("action"):
        print(f"  action: {result['action']}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
