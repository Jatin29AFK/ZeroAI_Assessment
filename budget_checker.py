"""Small helper for checking LiteLLM-compatible key budget information."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def print_guide() -> None:
    print(
        """
LiteLLM Budget Checker

Required .env values:
  LLM_API_BASE="https://recllm.brahmastra.tech/"
  LLM_MODEL_NAME=groq/openai/gpt-oss-120b
  LLM_API_KEY=your-api-key

Commands:
  python budget_checker.py
  python budget_checker.py --guide
""".strip()
    )


def get_key_info(api_key: str, base_url: str) -> dict:
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    if not base_url:
        raise RuntimeError("LLM_API_BASE is not set")

    endpoint = base_url.rstrip("/") + "/key/info"
    query = urllib.parse.urlencode({"key": api_key})
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc


def display_budget(info: dict) -> None:
    data = info.get("info") if isinstance(info.get("info"), dict) else info
    max_budget = data.get("max_budget")
    spend = float(data.get("spend", 0.0) or 0.0)
    user_id = data.get("user_id", "Unknown")

    print("\nAPI Budget Status")
    print("-----------------")
    print(f"User ID:     {user_id}")
    print(f"Total Spend: ${spend:.4f}")
    if max_budget is None:
        print("Max Budget:  Unlimited or unavailable")
    else:
        max_budget = float(max_budget)
        print(f"Max Budget:  ${max_budget:.4f}")
        print(f"Remaining:   ${max_budget - spend:.4f}")
    print("-----------------\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LiteLLM API key budget")
    parser.add_argument("--guide", action="store_true", help="Show usage guide and exit")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")

    if args.guide:
        print_guide()
        return 0

    try:
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_API_BASE", "")
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "(unset)"
        print(f"Checking budget for key: {masked}")
        display_budget(get_key_info(api_key, base_url))
        return 0
    except Exception as exc:
        print(f"Budget check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
