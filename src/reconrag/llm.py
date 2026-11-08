"""Multi-provider LLM client for live generation, Layer 3 verification, and judging."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from a ``.env`` file into a dictionary."""
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k not in env:
            env[k] = v
    return env


def call_openai_compatible(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 90,
) -> str:
    """Invoke an OpenAI-compatible ``/chat/completions`` endpoint."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ReconRAG/1.0",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data["choices"][0]["message"].get("content") or "").strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    return ""
